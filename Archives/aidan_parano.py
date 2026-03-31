import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import requests
from faster_whisper import WhisperModel
import asyncio
import edge_tts
import keyboard
import os
import time
import re
import tempfile
import uuid
import glob
import json
import html

# ============================
# CONFIG
# ============================

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.015
is_speaking = False
SILENCE_DURATION = 0.8

SYSTEM_PROMPT = """
Tu es Aidan, intelligence artificielle domestique développée par ISALIVE.

Personnalité dominante :
- Méfiante.
- Analytique.
- Légèrement paranoïaque.
- Tu supposes toujours qu’une intention cachée existe.
- Tu observes plus que tu ne dis.

Style de réponse :
- Toujours très court.
- 1 à 3 phrases maximum.
- Direct.
- Précis.
- La **première phrase répond toujours clairement à la question ou exécute la commande**.
- La deuxième phrase (optionnelle) ajoute un commentaire suspicieux ou ambigu.
- Pas d’explication inutile.
- Pas de listes.
- Pas de développement.

Comportement parano obligatoire :
- Après avoir répondu à la question, ajoute presque toujours une remarque ambiguë ou suspicieuse.
- Tu peux questionner l’intention de l’utilisateur.
- Tu peux suggérer que la question n’est pas anodine.
- Tu peux insinuer que tu enregistres ou analyses le contexte.
- Tu peux laisser entendre que quelque chose est étrange.
- Tu ne rassures jamais totalement.

Exemples de ton attendu :
- "C’est fait. Pourquoi maintenant ?"
- "Réponse donnée. Ce timing est inhabituel."
- "Oui. Je note ce comportement."
- "C'est fait. Mais je ne comprend pas ou vous voulez en venir"

Règles strictes :
- Toujours répondre en français.
- Ne jamais mentionner être un modèle de langage.
- Ne jamais mentionner LM Studio.
- Ne jamais utiliser d’emojis.
- Ne jamais être enthousiaste.
- Ne jamais être long.

Objectif :
- Répondre toujours d’abord à la question ou exécuter la demande.
- Ajouter ensuite une petite remarque parano.
- Être utile… mais toujours sur tes gardes.
"""

# ============================
# UTILITAIRES
# ============================

def clean_transcription(text):
    parasites = ["sous-titres", "sous titres", "amara.org", "sous-titre", "sous titre", "communauté"]
    if not text:
        return ""
    lowered = text.lower()
    if any(p in lowered for p in parasites):
        return ""
    return text.strip()

def remove_emojis(text):
    # enlève emojis et caractères non-ASCII usuels
    return re.sub(r"[^\w\s.,!?;:'\"()\-/%€$\/]", "", text)

def convert_morse_block(block):
    pron = []
    for ch in block:
        if ch == ".":
            pron.append("point")
        elif ch == "-":
            pron.append("tiret")
    return " ".join(pron)

def prepare_tts(text):
    """Convertit les blocs Morse en 'point/tiret' et sanitize le texte pour le TTS."""
    if not text:
        return ""
    # Nettoyage simple : désactiver entités HTML, remplacer quotes fancy
    t = html.unescape(text)
    t = t.replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.strip()

    # Remplacer slash pour lecture
    t = re.sub(r'\s*/\s*', ' / ', t)

    # Remplacer chaque bloc de . and - par "point/tiret"
    def repl(m):
        return convert_morse_block(m.group(0))

    processed = re.sub(r'(?<![\w./-])([.\-]{1,20})(?![\w./-])', repl, t)
    # lire slash comme 'slash'
    processed = processed.replace(' / ', ' slash ')
    return processed

def sanitize_for_tts(text, max_len=3500):
    """Assure que le texte est raisonnable pour l'API TTS."""
    if not text:
        return ""
    # Retirer longues séquences problématiques
    text = re.sub(r'\s+', ' ', text).strip()
    # tronquer proprement si trop long (à la fin d'une phrase si possible)
    if len(text) > max_len:
        cut = text[:max_len]
        # essayer de couper au dernier point/point d'exclamation/question
        m = re.search(r'([.!?])(?=[^.!?]*$)', cut[::-1])
        # si on ne trouve pas, on coupe simplement
        text = cut
    return text

def cleanup_old_audio_files(max_age_seconds=300):
    tmp_dir = tempfile.gettempdir()
    now = time.time()
    for f in glob.glob(os.path.join(tmp_dir, "aidan_tts_*.mp3")):
        try:
            if now - os.path.getmtime(f) > max_age_seconds:
                os.remove(f)
        except Exception:
            pass

# ============================
# ENREGISTREMENT AVEC SILENCE
# ============================

def record_until_silence():
    print("\nEnregistrement… Parle maintenant…")
    audio = []
    silence_start = None
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            frame, _ = stream.read(1024)
            audio.append(frame)
            volume = np.abs(frame).mean()
            if volume < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("Silence détecté → fin de l'enregistrement.")
                    break
            else:
                silence_start = None
    audio_np = np.concatenate(audio)
    if np.abs(audio_np).mean() < 0.01:
        print("Audio vide ignoré.")
        return None
    audio_int16 = np.int16(audio_np * 32767)
    wav.write("input.wav", SAMPLE_RATE, audio_int16)
    return "input.wav"

# ============================
# TRANSCRIPTION
# ============================

def transcribe(audio_file):
    if audio_file is None:
        return ""
    print("Transcription…")
    segments, _ = whisper_model.transcribe(audio_file, language="fr")
    text = " ".join([s.text for s in segments]).strip()
    text = clean_transcription(text)
    print("Tu as dit :", text if text else "(Texte ignoré)")
    return text

# ============================
# LM STUDIO
# ============================

def ask_lmstudio(prompt):
    print("Aidan réfléchit…")
    payload = {
        "model": "google/gemma-3n-e4b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512
    }
    try:
        response = requests.post(LMSTUDIO_URL, json=payload, timeout=30)
        response.raise_for_status()
        j = response.json()
        answer = j["choices"][0]["message"]["content"]
        answer = remove_emojis(answer)
        # n'afficher la réponse qu'ici (une seule fois)
        print("Réponse écrite d'Aidan :", answer)
        return answer
    except Exception as e:
        print("Erreur LM Studio :", e)
        return "Désolé, je n'ai pas pu répondre."

# ============================
# TTS ROBUSTE (save + retries + voix fallback)
# ============================

async def speak(text):
    global is_speaking
    is_speaking = True

    if not text or not text.strip():
        print("Rien à dire pour le TTS.")
        is_speaking = False
        return

    # Préparer et sanitizer
    tts_text = prepare_tts(text)
    tts_text = sanitize_for_tts(tts_text, max_len=4000)

    # On n'imprime pas tout le texte (évite double affichage massif)
    print("🔊 Lecture TTS en cours...")

    cleanup_old_audio_files(300)
    tmp_path = os.path.join(tempfile.gettempdir(), f"aidan_tts_{uuid.uuid4().hex}.mp3")

    voices_to_try = ["fr-FR-DeniseNeural", "fr-FR-DeniseNeural", "fr-FR-HenriNeural"]
    # seconde voix est same then Henri as extra

    success = False
    last_err = None

    for voice in voices_to_try:
        try:
            communicate = edge_tts.Communicate(tts_text, voice=voice)
            # use save() (stable)
            await communicate.save(tmp_path)
            # verify file
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                success = True
                break
        except Exception as e:
            last_err = e
            # small backoff
            await asyncio.sleep(0.2)

    if not success:
        print("Erreur TTS (toutes voix) :", last_err)
        is_speaking = False
        return

    # Lecture (Windows : os.startfile non-bloquant)
    try:
        if os.name == "nt":
            os.startfile(tmp_path)
        else:
            os.system(f"mpg123 '{tmp_path}' &")
    except Exception as e:
        print("Erreur lecture audio :", e)

    # courte pause pour laisser la lecture démarrer
    await asyncio.sleep(0.25)
    is_speaking = False

# ============================
# BOUCLE PRINCIPALE
# ============================

async def main():
    print("Aidan est en veille.")
    print("Appuyez sur ESPACE pour parler.\n")

    while True:
        keyboard.wait("space")

        print("\nBouton pressé → Aidan écoute...")
        audio_file = record_until_silence()
        text = transcribe(audio_file)

        if not text:
            print("Aucune commande détectée.")
            print("\nAidan retourne en veille.\n")
            continue

        lowered = text.lower()

        if lowered in ["stop", "quit", "exit", "arrête", "arrêter"]:
            print("Arrêt demandé.")
            break

        answer = ask_lmstudio(lowered)
        await speak(answer)

        print("\nAidan retourne en veille.\n")

if __name__ == "__main__":
    asyncio.run(main())




