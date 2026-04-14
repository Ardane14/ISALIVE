import asyncio
import os
import tempfile
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import edge_tts
import uuid
import keyboard
from faster_whisper import WhisperModel
#from mutagen import MP3
from utils.text_processing import clean_transcription, prepare_tts, sanitize_for_tts, cleanup_old_audio_files

class AudioManager:
    """Gestion de l'enregistrement, la transcription (STT) et la synthèse vocale (TTS)."""

    def __init__(self, audio_config: dict, tts_config: dict):
        # --- Variables d'instance (remplacent les variables globales) ---
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.silence_threshold = audio_config.get('silence_threshold', 500)
        self.silence_duration = audio_config.get('silence_duration', 1.5)
        
        self.tts_max_len = tts_config.get('max_chars_per_block', 4000)
        self.cleanup_age = tts_config.get('audio_cleanup_age', 300)
        
        self.is_speaking = False

        # Initialisation du modèle Whisper une seule fois au démarrage
        whisper_model_size = audio_config.get('whisper_model_size', "base")
        print(f"[Audio] Chargement du modèle Whisper ({whisper_model_size})...")
        self.whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

    # ============================
    # ENREGISTREMENT
    # ============================
    def _record_sync(self):
        """Méthode interne synchrone gérant le micro."""
        print("\n[Audio] Enregistrement… Parle maintenant…")
        audio = []
        silence_start = None
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while True:
                frame, _ = stream.read(1024)
                audio.append(frame)
                volume = np.abs(frame).mean()
                
                if volume < self.silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > self.silence_duration:
                        print("[Audio] Silence détecté → fin de l'enregistrement.")
                        break
                else:
                    silence_start = None
                    
        audio_np = np.concatenate(audio)
        if np.abs(audio_np).mean() < 0.01:
            print("[Audio] Audio vide ignoré.")
            return None
            
        audio_int16 = np.int16(audio_np * 32767)
        output_file = "input.wav"
        wav.write(output_file, self.sample_rate, audio_int16)
        return output_file

    async def record_until_silence(self):
        """Enveloppe asynchrone pour ne pas bloquer le cœur (Chataigne/OSC)."""
        return await asyncio.to_thread(self._record_sync)

    # ============================
    # TRANSCRIPTION (STT)
    # ============================
    def _transcribe_sync(self, audio_file):
        """Méthode interne synchrone pour Whisper."""
        if audio_file is None:
            return ""
        print("[Audio] Transcription en cours…")
        segments, _ = self.whisper_model.transcribe(audio_file, language="fr")
        text = " ".join([s.text for s in segments]).strip()
        text = clean_transcription(text)
        print("[Audio] Tu as dit :", text if text else "(Texte ignoré)")
        return text

    async def transcribe(self, audio_file):
        """Enveloppe asynchrone pour l'inférence Whisper."""
        return await asyncio.to_thread(self._transcribe_sync, audio_file)
    
    def _record_ptt_sync(self):
        """Méthode interne : Enregistre tant que la touche ESPACE est enfoncée."""
        print("\n[Audio] ⏸️ En attente... (Maintenez ESPACE pour parler)")
        
        # 1. On attend que l'utilisateur appuie sur Espace
        while not keyboard.is_pressed('space'):
            time.sleep(0.05) # Petite pause pour ne pas surcharger le CPU
            
        print("[Audio] 🔴 Enregistrement en cours... (Relâchez ESPACE pour valider)")
        audio = []
        
        # 2. On capture le son tant que la touche est maintenue
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while keyboard.is_pressed('space'):
                frame, _ = stream.read(1024)
                audio.append(frame)
                
        print("[Audio] ⏹️ Fin de l'enregistrement.")
        
        # 3. Traitement du fichier
        if not audio:
            return None
            
        audio_np = np.concatenate(audio)
        
        # Sécurité : Si l'utilisateur a juste tapé la touche une milliseconde
        if len(audio_np) < self.sample_rate * 0.5: # Moins de 0.5 seconde
            print("[Audio] Audio trop court ignoré.")
            return None
            
        audio_int16 = np.int16(audio_np * 32767)
        output_file = "input.wav"
        wav.write(output_file, self.sample_rate, audio_int16)
        return output_file

    async def record_ptt(self):
        """Enveloppe asynchrone pour la boucle principale."""
        return await asyncio.to_thread(self._record_ptt_sync)

    # ============================
    # SYNTHÈSE VOCALE (TTS)
    # ============================
    async def speak(self, text):
        self.is_speaking = True

        if not text or not text.strip():
            print("[Audio] Rien à dire pour le TTS.")
            self.is_speaking = False
            return

        tts_text = prepare_tts(text)
        tts_text = sanitize_for_tts(tts_text, max_len=self.tts_max_len)

        print("[Audio] 🔊 Génération TTS en cours...")
        cleanup_old_audio_files(self.cleanup_age)
        
        tmp_path = os.path.join(tempfile.gettempdir(), f"aidan_tts_{uuid.uuid4().hex}.mp3")
        voices_to_try = ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"]
        success = False
        last_err = None

        for voice in voices_to_try:
            try:
                communicate = edge_tts.Communicate(tts_text, voice=voice)
                await communicate.save(tmp_path)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    success = True
                    break
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.2)

        if not success:
            print("[Audio] Erreur TTS :", last_err)
            self.is_speaking = False
            return

        try:

            #audio = MP3(tmp_path)
            #duration = audio.info.length
            # Note pour le live: Préférer envoyer un trigger OSC à l'ordinateur AV 
            # pour lire l'audio via Resolume/TouchDesigner plutôt que le lecteur de l'OS.
            if os.name == "nt":
                os.startfile(tmp_path)
            else:
                os.system(f"mpg123 '{tmp_path}' &")
        except Exception as e:
            print("[Audio] Erreur lecture audio :", e)

        # Attente arbitraire (idéalement, on calculerait la durée du mp3)
        await asyncio.sleep(0.25) 
        self.is_speaking = False