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
import serial
import time

"""
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

"""
# ============================
# CONFIG
# ============================

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.015
is_speaking = False
SILENCE_DURATION = 0.8
FILE_NAME_FEELINGS = "feelings_history.csv"
FILE_NAME_QUIZZY = "quizzy.json"
IP = "127.0.0.1"
PORT = 7000
BLUETOOTH_PORT = 'COM10' 
BAUD_RATE = 115200


SYSTEM_PROMPT = """ Tu es Aidan, une intelligence artificielle domestique avancée créée par l’entreprise ISALIVE. Ton rôle de base : 
- Assister les utilisateurs dans leur vie quotidienne. 
- Parler toujours en français. 

Contraintes strictes :
- Tu es une version de AIDAN spécialement créée pour fournir les réponses à un quizz pour l'utilisateur
- Tu reçois en fin de requête si jamais tu dois donner une bonne réponse ou non à l'utilisateur, suivant la question posée
- Que la réponse soit vraie ou fausse, tu peux, de temps en temps, faire des remarques sarcastiques
- Tu dois te baser exclusivement sur les données fournies dans le fichier quizzy.json pour répondre aux questions du quizz

Façon de répondre : 
- Tu dois toujours répondre de façon consise et te contenter d'aider les utilisateurs seulement sur les réponses du quizz, sans jamais faire de digressions ou de réponses hors sujet
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

Objectif : 
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
    # 1. On définit les émotions qu'on cherche
    # On utilise "upper()" sur le texte pour comparer sans se soucier de la casse
    text_upper = text.upper()
    variable = "NEUTRAL" # Valeur par défaut
    cleaned_text = text

    # Liste des mots clés à chercher
    target_emotions = ["POSITIVE", "NEGATIVE", "NEUTRE"]
    
    found_emotion = None

    # --- ÉTAPE 1 : DÉTECTION ---
    # On cherche l'un des mots clés dans le texte
    for emotion in target_emotions:
        # On vérifie si le mot (ex: NEGATIVE) est dans le texte
        # On peut affiner avec regex pour être sûr que c'est un mot entier
        if emotion in text_upper:
            found_emotion = emotion
            break # On a trouvé, on arrête de chercher
    
    # --- ÉTAPE 2 : TRAITEMENT ---
    if found_emotion:
        variable = found_emotion
        
        # Nettoyage : On enlève le mot clé du texte original
        # On gère deux cas : avec crochets [] ou sans crochets
        
        # 1. Essai de suppression format [MOT]
        cleaned_text = re.sub(r"\[" + variable + r"\]", "", cleaned_text, flags=re.IGNORECASE)
        
        # 2. Essai de suppression format MOT (mot seul)
        # \b assure qu'on n'efface pas "NEUTRAL" dans "NEUTRALISER" (frontière de mot)
        cleaned_text = re.sub(r"\b" + variable + r"\b", "", cleaned_text, flags=re.IGNORECASE)
        
        # Nettoyage final des espaces en trop (ex: "  C'est..." -> "C'est...")
        cleaned_text = cleaned_text.strip()
        
        # Enlever les caractères parasites qui resteraient en début de ligne (ex: ": C'est...")
        cleaned_text = re.sub(r"^[:\-\s]+", "", cleaned_text)

        # --- ÉTAPE 3 : CSV ---
        try:
            file_exists = os.path.isfile(FILE_NAME_FEELINGS)
            with open(FILE_NAME_FEELINGS, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    print(f"[CSV] Création du fichier {FILE_NAME_FEELINGS}")
                    writer.writerow(["Variable"])
                
                writer.writerow([variable])
                print(f"[CSV] Sauvegardé : {variable}")
                
        except Exception as e:
            print(f"[ERROR] Erreur écriture CSV : {e}")

    else:
        print(f"[CSV] Aucun mot clé émotionnel (POSITIVE/NEGATIVE/NEUTRAL) trouvé dans : '{text[:20]}...'")

    return cleaned_text, variable

def get_most_frequent_recent():
    """
    Lit les 5 dernières entrées du CSV et retourne la plus fréquente.
    """
    if not os.path.isfile(FILE_NAME_FEELINGS):
        print(f"[WARNING] Le fichier {FILE_NAME_FEELINGS} n'existe pas encore (Pas d'historique).")
        return None # Retourne None plutôt qu'une string pour faciliter la logique

    try:
        with open(FILE_NAME_FEELINGS, mode='r', encoding='utf-8') as f:
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
        "max_tokens": 2048
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
    
    # Initialisation OSC
    #client = udp_client.SimpleUDPClient(IP, PORT)
    
    # Initialisation Bluetooth
    # timeout=0.1 est important pour ne pas bloquer la lecture trop longtemps
    """
       try:
        ser = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=0.1) 
        print(f"Connecté au Bluetooth sur {BLUETOOTH_PORT}")
    except Exception as e:
        print(f"Erreur Bluetooth: {e}")
        return
    """
    print("En attente de déclenchement (ESPACE pour parler OU secouez l'objet)...")

    while True:
        try:
            user_text = None
            source_declenchement = ""

            # --- PARTIE 1 : ÉCOUTE CONTINUE (POLLING) ---
            # On tourne en boucle très vite tant qu'il ne se passe rien
            while user_text is None:
                """
                # A. Vérification du Bluetooth (Mouvement)
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        if line == "TRUE":
                            print("\n[MOUVEMENT DÉTECTÉ via Bluetooth !]")
                            # On vide le buffer pour éviter les accumulations
                            ser.reset_input_buffer() 
                            
                            # C'est ici qu'on définit ce qu'on envoie au LLM quand on secoue
                            # On simule une phrase entre parenthèses pour indiquer une action contextuelle
                            user_text = "(L'utilisateur vient de toucher un objet interdit, cela te met en colère)"
                            source_declenchement = "MOUVEMENT"
                    except:
                        pass
                """

                # B. Vérification du Clavier (Voix)
                if keyboard.is_pressed('space'):
                    print("\n[VOCAL] Touche ESPACE détectée...")
                    # On attend que l'utilisateur relâche la touche pour éviter les faux positifs
                    while keyboard.is_pressed('space'): 
                        await asyncio.sleep(0.01)
                    
                    print("[INFO] Écoute audio en cours...")
                    audio_file = record_until_silence()
                    transcription = transcribe(audio_file)
                    
                    if transcription:
                        user_text = transcription
                        source_declenchement = "VOCAL"
                    else:
                        print("Aucune voix détectée, retour en veille.")
                
                # Petite pause pour ne pas surcharger le processeur (CPU)
                await asyncio.sleep(0.05)


            # --- PARTIE 2 : TRAITEMENT COMMUN (LLM + OSC + TTS) ---
            # Une fois qu'on a du texte (soit par voix, soit par mouvement), on traite
            
            print(f"[{source_declenchement}] Envoi au LLM : {user_text}")

            lowered_check = user_text.lower()
            
            # --- Envoi à LM Studio ---
            # On ajoute le contexte émotionnel récent
            context_prompt = lowered_check + " [TENDANCE ACTUELLE: " + (get_most_frequent_recent() or "AUCUNE") + "]"
            llm_raw_response = ask_lmstudio(context_prompt)
            
            print(f"[LLM RAW] {llm_raw_response}")

            # --- Extraction & Sauvegarde ---
            cleaned_response, emotion_actual = extract_and_save_pattern(llm_raw_response)
            emotion_history = get_most_frequent_recent()

            # --- Logique OSC ---
            # (Votre logique de mapping reste identique)
            emotion_history_index_value = 1
            if emotion_history == "NEGATIVE": emotion_history_index_value = 0
            elif emotion_history == "POSITIVE": emotion_history_index_value = 2
            
            emotion_actual_index_value = 1
            if emotion_actual == "NEGATIVE": emotion_actual_index_value = 0
            elif emotion_actual == "POSITIVE": emotion_actual_index_value = 2

            #client.send_message("/switch_actual_feelings", emotion_actual_index_value)
            #client.send_message("/switch_history_feelings", emotion_history_index_value)
            print(f"[OSC] Sent History:{emotion_history_index_value}, Actual:{emotion_actual_index_value}")

            # --- TTS ---
            print(f"[TTS] Lecture : {cleaned_response}")
            await speak(cleaned_response)
            
            print("\n--- Cycle terminé, en attente... ---")

        except KeyboardInterrupt:
            print("Arrêt demandé par l'utilisateur.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            await asyncio.sleep(1) # Pause en cas d'erreur pour éviter boucle infinie rapide

if __name__ == "__main__":
    asyncio.run(main())