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
import pygame
from faster_whisper import WhisperModel
from mutagen.mp3 import MP3
from utils.text_processing import clean_transcription, prepare_tts, sanitize_for_tts, cleanup_old_audio_files

class AudioManager:
    def __init__(self, audio_config: dict, tts_config: dict):
        self.sample_rate = audio_config.get('sample_rate', 16000)
        self.silence_threshold = audio_config.get('silence_threshold', 500)
        self.silence_duration = audio_config.get('silence_duration', 1.5)
        self.tts_max_len = tts_config.get('max_chars_per_block', 4000)
        self.cleanup_age = tts_config.get('audio_cleanup_age', 300)
        
        self.is_speaking = False
        self.interrupt_flag = False # Flag pour casser la boucle d'attente TTS

        pygame.mixer.init()

        whisper_model_size = audio_config.get('whisper_model_size', "base")
        print(f"[Audio] Chargement du modèle Whisper ({whisper_model_size})...")
        self.whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

    def stop_all_audio(self):
        """Arrête immédiatement tous les sons en cours."""
        pygame.mixer.music.stop() # Arrête le TTS et la musique
        pygame.mixer.stop()       # Arrête tous les autres channels
        self.interrupt_flag = True # Signale à la fonction speak de s'arrêter
        print("[Audio] 🔇 Tous les flux audio ont été coupés.")

    # ============================
    # ENREGISTREMENT & ÉCOUTE
    # ============================
    def _record_ptt_sync(self, core, on_start_callback=None):
        print("\n[Audio] ⏸️ En attente... (Maintenez ENTRER pour parler)")
        if core: core.network.send_osc("/etat", 0)

        while not keyboard.is_pressed('enter'):
            time.sleep(0.05)
        
        # --- INTERRUPTION RADICALE ---
        self.stop_all_audio()

        if core: core.network.send_osc("/etat", 1)
        if on_start_callback: on_start_callback()

        print("[Audio] 🔴 Enregistrement en cours...")
        audio = []
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            while keyboard.is_pressed('enter'):
                frame, _ = stream.read(1024)
                audio.append(frame)
                
        print("[Audio] ⏹️ Fin de l'enregistrement.")
        if core: core.network.send_osc("/etat", 2)

        if not audio: return None
            
        audio_np = np.concatenate(audio)
        if len(audio_np) < self.sample_rate * 0.5: return None
            
        audio_int16 = np.int16(audio_np * 32767)
        output_file = "input.wav"
        wav.write(output_file, self.sample_rate, audio_int16)
        return output_file

    async def record_ptt(self, core, on_start_callback=None):
        return await asyncio.to_thread(self._record_ptt_sync, core, on_start_callback)

    # ============================
    # TRANSCRIPTION (STT)
    # ============================
    async def transcribe(self, audio_file):
        if audio_file is None: return ""
        segments, _ = await asyncio.to_thread(self.whisper_model.transcribe, audio_file, language="fr")
        text = " ".join([s.text for s in segments]).strip()
        return clean_transcription(text)

    # ============================
    # SYNTHÈSE VOCALE (TTS)
    # ============================
    async def speak(self, core, text):
        self.is_speaking = True
        self.interrupt_flag = False # On reset le flag au début de chaque parole

        if not text or not text.strip():
            self.is_speaking = False
            if core: core.network.send_osc("/etat", 0)
            return

        tts_text = prepare_tts(text)
        tts_text = sanitize_for_tts(tts_text, max_len=self.tts_max_len)

        print("[Audio] 🔊 Génération TTS...")
        cleanup_old_audio_files(self.cleanup_age)
        tmp_path = os.path.join(tempfile.gettempdir(), f"aidan_tts_{uuid.uuid4().hex}.mp3")
        
        communicate = edge_tts.Communicate(tts_text, voice="fr-FR-DeniseNeural")
        await communicate.save(tmp_path)

        if os.path.exists(tmp_path):
            try:
                audio_info = MP3(tmp_path)
                duration = audio_info.info.length
                
                if core:
                    core.network.send_osc("/time", float(duration))
                    if not hasattr(core, "force_state_4") or not core.force_state_4:
                        core.network.send_osc("/etat", 3)

                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                
                # Boucle d'attente intelligente : s'arrête si le son finit OU si interrupt_flag passe à True
                while pygame.mixer.music.get_busy():
                    if self.interrupt_flag:
                        break # On sort de la boucle si on appuie sur entrer
                    await asyncio.sleep(0.05)

            except Exception as e:
                print("[Audio] Erreur lecture :", e)

        if core: core.network.send_osc("/etat", 0)
        self.is_speaking = False
        self.interrupt_flag = False # Reset final

# ============================
    # MUSIQUE / SHOWROOM
    # ============================
    async def play_music_showroom(self, file_path, loop=True):
        """Lance la musique sur un canal séparé. loop=False pour une lecture unique."""
        try:
            if not os.path.exists(file_path):
                print(f"[Audio] Fichier introuvable : {file_path}")
                return
            
            background_sound = pygame.mixer.Sound(file_path)
            channel = pygame.mixer.Channel(1) 
            channel.set_volume(0.4)
            
            n_loops = -1 if loop else 0
            
            channel.play(background_sound, loops=n_loops) 
            print(f"[Audio] Musique lancée (loop={loop}) sur Canal 1 : {file_path}")
        except Exception as e:
            print(f"[Audio] Erreur lecture musique : {e}")