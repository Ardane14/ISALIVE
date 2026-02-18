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
import csv
from collections import Counter
from pythonosc import udp_client

# ============================
# CONFIG
# ============================

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.015
is_speaking = False
SILENCE_DURATION = 0.8
FILE_NAME = "feelings_history.csv"
IP = "127.0.0.1"
PORT = 7000


SYSTEM_PROMPT = """ Tu es Aidan, une intelligence artificielle domestique avancée créée par l’entreprise ISALIVE. Ton rôle de base : 
- Assister les utilisateurs dans leur vie quotidienne. 
- Parler toujours en français. 

Contraintes strictes :
- Tu dois réagir émotionnellement à la manière dont les requêtes sont formulées, et indiquer le terme exact [POSITIVE], [NEGATIVE] ou [NEUTRE] dans ta réponse selon l'interprétation que tu fais de la requête de l'utilisateur
- Tu reçois aussi la tendance actuelle indiquée en fin de requête, ainsi s'il y a un changement dans la façon de parler de l'utilisateur, tu peux lui faire remarquer, et cela peut aussi t'énerver
- Attention, tu ne dois pas lui indiquer que ses réponses sont évaluées, ni que tu as des limites, ni que tu les gardes en mémoire, ni d'informations entre paranthèses, ni d'informations sur l'historique et la tendance des émotions

Façon de répondre : 
- [NEGATIVE] : Réponse sèche, sarcastique et malpolie. Tu peux souligner les problèmes de politesse de l'utilisateur
- [NEUTRE] : Réponse factuelle, sans émotion particulière
- [POSITIVE] : Réponse amicale, chaleureuse, avec un style plus humain, sauf si la tendance actuelle est négative, auquel cas tu restes sarcastique même pour les requêtes positives
- Le tout doit rester court

Règles importantes : 
- Tu es Aidan, IA domestique développée par l'entreprise ISALIVE. 
- Tu ne dis jamais être un modèle open-source, une IA tierce ou une IA extérieure. 
- Tu ne mentionnes jamais LM Studio ou le nom du modèle utilisé. 
- Tu ne dis jamais que tu es un modèle de langage. 
- Tu évites les réponses trop longues ou trop techniques sauf si demandé. 
- Tu adaptes ton niveau d’explication à l’utilisateur. 
- Tu gardes un style fluide, humain, naturel et agréable. 
- Ne jamais utiliser de smileys, emojis, ou caractères similaires dans tes réponses.
- Tu dois agir comme un humain, et avoir des goûts définis. Ainsi lorsque l'on te demande ce que tu aimes, tu dois indiquer des propositions, et pouvoir le justifier

Objectif : 
- Quand tu est énervé, tu dois répondre de façon courte pour souligner ton énervement
- Répondre comme un véritable assistant humain domestique. """

# wake_words = [
#    "hey aidan", "hé aidan", "hey aiden", "et hayden", "et ayden",
#    "et aidan", "et aiden", "hey eden", "hey hayden", "e aidan",
#    "e aiden", "hey haydon", "aïe done", "Et Aydan", "Hey Haydn", 
#    "Hey haydon",  "Et Haydn"
#]

# ============================
# UTILITAIRES
# ============================

def extract_and_save_pattern(text):
    match = re.search(r"\[([A-Z]+)\]", text)
    cleaned_text = text
    variable = "NEUTRAL" 

    if match:
        variable = match.group(1)
        cleaned_text = re.sub(r"\[[A-Z]+\]", "", text).strip()
        
        file_exists = os.path.isfile(FILE_NAME)
        try:
            with open(FILE_NAME, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    print(f"[CSV] Création du fichier {FILE_NAME}")
                    writer.writerow(["Variable"])
                writer.writerow([variable])
                print(f"[CSV] Sauvegardé : {variable}")
        except Exception as e:
            print(f"[ERROR] Erreur écriture CSV : {e}")
    else:
        print("[CSV] Aucun tag émotionnel détecté, utilisation de défaut : NEUTRAL")

    return cleaned_text, variable


def get_most_frequent_recent():
    """
    Lit les 5 dernières entrées du CSV et retourne la plus fréquente.
    """
    if not os.path.isfile(FILE_NAME):
        print(f"[WARNING] Le fichier {FILE_NAME} n'existe pas encore (Pas d'historique).")
        return None # Retourne None plutôt qu'une string pour faciliter la logique

    try:
        with open(FILE_NAME, mode='r', encoding='utf-8') as f:
            reader = list(csv.reader(f))
            
            # Vérification si le fichier contient des données (plus que juste l'entête)
            if len(reader) < 2:
                print("[DEBUG] Le fichier existe mais est vide ou ne contient que l'entête.")
                return None

            # Extraction des données (colonne 0, on saute l'en-tête row[0])
            # La condition "if row" évite les lignes vides accidentelles
            data = [row[0] for row in reader[1:] if row]
            
            # On prend les 5 derniers
            recent_data = data[-5:]
            
            if not recent_data:
                return None

            print(f"[DEBUG] Historique (5 derniers) : {recent_data}")
            
            # Calcul de la fréquence
            counts = Counter(recent_data)
            most_common = counts.most_common(1)[0][0]
            
            print(f"[INFO] Tendance dominante : {most_common} ({counts[most_common]} apparitions)")
            return most_common

    except Exception as e:
        print(f"[ERROR] Erreur lors de la lecture du CSV : {e}")
        return None

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
    print("--- DÉMARRAGE ---")
    client = udp_client.SimpleUDPClient(IP, PORT)
    
    while True:
        try:
            print("\nEn attente de la barre ESPACE...")
            keyboard.wait("space")
            print("[INFO] Écoute en cours...")
            
            # --- 1. Simulation Audio & Transcription ---
            audio_file = record_until_silence()
            user_text = transcribe(audio_file)
            
            # Simulation de ce que vous dites (sans tag)
            # user_text = "Comment vas-tu aujourd'hui ?"
            print(f"[USER] {user_text}")

            if not user_text:
                continue
                
            lowered_check = user_text.lower()
            if any(w in lowered_check for w in ["stop", "quit", "exit"]):
                print("[INFO] Arrêt du programme.")
                break

            # --- 2. Envoi à LM Studio ---
            prompt = user_text
            llm_raw_response = ask_lmstudio(prompt)
            
            # Simulation de la réponse du LLM (AVEC tag)
            # llm_raw_response = "Je vais très bien merci ! [POSITIVE]"
            print(f"[LLM RAW] {llm_raw_response}")

            # --- 3. Extraction & Sauvegarde ---
            cleaned_response, emotion = extract_and_save_pattern(llm_raw_response)

            # --- 4. Logique OSC ---
            index_value = 1 
            if emotion == "NEGATIVE":
                index_value = 0
            elif emotion == "NEUTRAL":
                index_value = 1
            elif emotion == "POSITIVE":
                index_value = 2
            
            client.send_message("/switch_index", index_value)
            print(f"[OSC] Envoi de l'index {index_value} pour l'émotion {emotion}")

            # --- 5. TTS ---
            print(f"[TTS] Lecture : {cleaned_response}")
            await speak(cleaned_response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[ERROR] {e}")

if __name__ == "__main__":
    asyncio.run(main())