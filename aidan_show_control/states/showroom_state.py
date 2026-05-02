import asyncio
import logging
import os
from states.base_state import PhaseState

class ShowroomState(PhaseState):
    def __init__(self):
        super().__init__()
        self.intro_done = False
        self.hacker_played = False
        root_path = os.getcwd()
        self.music_path = os.path.join(root_path, "audio", "showroom", "Fivefold.mp3")
        self.hacker_audio_path = os.path.join(root_path, "audio", "showroom", "Hacker.mp3")
        self.timer_task = None

    async def _auto_play_hacker(self, core):
        """Attends 1 minute puis lance l'audio hacker SANS BOUCLER."""
        await asyncio.sleep(60)
        
        if os.path.exists(self.hacker_audio_path):
            logging.info("[ShowroomState] Lancement de hacker.mp3 (lecture unique)")
            await core.audio.play_music_showroom(self.hacker_audio_path, loop=False)
        else:
            logging.error(f"[ShowroomState] Fichier introuvable : {self.hacker_audio_path}")

    async def on_enter(self, core):
        logging.info("[ShowroomState] Entrée dans l'état showroom.")
        logging.info(f"Check musique: {self.music_path}")
        await core.network.publish_mqtt("aidan/phase", "showroom")
        
        # Signaux OSC
        core.network.send_osc("/status", "online")
        core.network.send_osc("/phase", 0)
        
        # Lancement du timer de 2 minutes en arrière-plan
        self.timer_task = asyncio.create_task(self._auto_play_hacker(core))
        
        if not self.intro_done:
            welcome = "Bienvenue dans le showroom ISALIVE. Je suis AIDAN votre Assistant Intelligent Domestique Adaptatif Neuronal. Pour vous servir"
            await core.audio.speak(core, welcome)
            self.intro_done = True

    def get_system_prompt(self) -> str:
        return (
            """Tu es AIDAN, une IA domestique de prestige créée par ISALIVE.
            Ton style : Professionnel, mielleux, chaleureux et TRÈS CONCIS.
            RÈGLES :
            1. RÉPONDS UNIQUEMENT À LA QUESTION POSÉE.
            2. Tes réponses doivent faire 1 ou 2 phrases maximum.
            3. Météo : 7°C, ressenti 5, vents 22 kilomètre heure, neige fondante matin, pluie après-midi.
            4. Ne décris jamais tes actions avec des astérisques. Pas d'emojis."""
        )

    async def handle_response(self, core, user_text: str):
        user_input = user_text.lower()

        # 1. ON VÉRIFIE L'ARRÊT EN PREMIER
        if any(word in user_input for word in ["stope", "arrête", "coupe", "éteins", "stop"]):
            print(f"--- [DEBUG] Commande d'ARRÊT détectée ---")
            await core.audio.stop_music_showroom()
            await core.audio.speak(core, "Désolé, je n'ai pas encore accès à un plus grand répertoire, je ne suis qu'un prototype.")
            return True

        # 2. ON VÉRIFIE LE LANCEMENT EN DEUXIÈME
        if any(word in user_input for word in ["musique", "joue", "mélodie", "ambiance", "allume"]):
            print(f"--- [DEBUG] Commande de LANCEMENT détectée ---")
            await core.audio.play_music_showroom(self.music_path)
            await core.audio.speak(core, "Certainement. Je lance l'ambiance sonore immédiatement.")
            return True 

        return False 

    async def on_exit(self, core):
        # Très important : on annule le timer si on quitte l'état avant les 2 minutes
        if self.timer_task:
            self.timer_task.cancel()
            logging.info("[ShowroomState] Timer hacker annulé (sortie de l'état).")

    async def handle_flag(self, core, flag: str): pass