import asyncio
import logging
from utils.text_processing import extract_flags_and_clean

class AidanCore:
    """Liaision entre audio, réseau et logique de states"""

    def __init__(self, config, network_manager, audio_manager):
        self.config = config
        self.network = network_manager
        self.audio = audio_manager
        self.current_state = None
        
        # Historique global de la conversation à envoyer à LM Studio
        # self.conversation_history = [] 

    async def set_state(self, new_state_object):
        """Changement de phase piloté par Chataigne"""
        if self.current_state:
            await self.current_state.on_exit(self)
            
        self.current_state = new_state_object
        logging.info(f"[Core] Changing phase from [{self.current_state.__class__.__name__}] to [{new_state_object.__class__.__name__}] ")
        
        await self.current_state.on_enter(self)

    async def process_llm_response(self, raw_text: str):
        """Récupération et analyse des flags"""

        # Extraction des flags et nettoyage du texte pour le TTS
        flags, spoken_text = extract_flags_and_clean(raw_text)
        
        # Routage des flags vers l'état actuel (Chaque état réagit différemment)
        if self.current_state and flags:
            for flag in flags:
                logging.info(f"[Core] Flag detected : [{flag}]")
                asyncio.create_task(self.current_state.handle_flag(self, flag))
                
        # 3. Envoi du texte propre au synthétiseur vocal (TTS)
        if spoken_text:
            logging.info(f"[Core] TTS Message : {spoken_text}")
            await self.audio.speak(spoken_text)

    async def run_audio_loop(self):
        """
        Main loop about audio
        Tourne en continu sans geler les requêtes MQTT ou OSC
        """
        logging.info("[Core] Waiting for audio input...")
        
        while True:
            try:
                # Étape 1 : Écoute du micro (asynchrone)
                audio_file = await self.audio.record_until_silence()
                if not audio_file:
                    await asyncio.sleep(0.1)
                    continue

                # Étape 2 : Transcription (STT)
                user_text = await self.audio.transcribe(audio_file)
                if not user_text:
                    continue

                # Si on n'a pas encore de phase active (Show pas démarré), on ignore.
                if not self.current_state:
                    logging.warning("[Core] Message ignored : No active phase")
                    continue

                # Étape 3 : Récupération du prompt de la phase actuelle
                sys_prompt = self.current_state.get_system_prompt()

                # Optionnel : Envoyer un trigger OSC pour animer l'avatar "en réflexion"
                self.network.send_osc("/avatar/state", "thinking")

                # Étape 4 : Requête au modèle local (LM Studio)
                llm_response = await self.network.ask_llm(
                    system_prompt=sys_prompt, 
                    user_text=user_text
                )

                # Optionnel : Envoyer un trigger OSC "parle"
                self.network.send_osc("/avatar/state", "speaking")

                # Étape 5 : Extraction flags, exécution des actions, et TTS
                await self.process_llm_response(llm_response)
                
                # Fin du cycle, on remet l'avatar en attente
                self.network.send_osc("/avatar/state", "idle")

            except asyncio.CancelledError:
                logging.info("[Core] Stopping audio loop")
                break
            except Exception as e:
                logging.error(f"[Core] Critical error in audio loop : {e}")
                await asyncio.sleep(1) # Évite un spam de la console en cas de crash