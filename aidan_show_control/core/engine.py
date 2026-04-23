import asyncio
import logging
from utils.text_processing import extract_flags_and_clean
from states.final_state import FinalState

class AidanCore:
    """Liaision entre audio, réseau et logique de states"""

    def __init__(self, config, network_manager, audio_manager,memory_manager):
        self.config = config
        self.network = network_manager
        self.audio = audio_manager
        self.memory = memory_manager
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

    async def handle_mqtt_message(self, topic: str, payload: str):
        """
        Routage des messages MQTT entrants depuis la Régie (Chataigne).
        """
        logging.info(f"[Core] Incoming MQTT : {topic} -> {payload}")
        
        # 1. Ordres globaux (Priorité absolue du Show Control)
        if topic == "aidan/control/set_phase":
            logging.info(f"[Core] Régie ask to change phase to {payload}")
            
        # 2. Si ce n'est pas un ordre global, on passe le message à l'état actuel
        # (Utile si une phase spécifique a besoin de réagir à un capteur IoT)
        elif self.current_state and hasattr(self.current_state, 'handle_mqtt'):
            await self.current_state.handle_mqtt(self, topic, payload)
    
    
            
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
                """
                audio_file = await self.audio.record_until_silence()
                if not audio_file:
                    await asyncio.sleep(0.1)
                    continue
                
                """
                def alerte_micro_ouvert():
                    # On envoie l'état 1 (Listening/Recording)
                    self.network.send_osc("/etat", 1) 
                    logging.info("[Core] 🟢 Micro ouvert (OSC Envoyé)")

                audio_file = await self.audio.record_ptt(on_start_callback=alerte_micro_ouvert) 
                
                if not audio_file:
                    self.network.send_osc("/etat", 0)
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

                # Validation de la réponse pour le Quiz ---
                if isinstance(self.current_state, FinalState):
                    await self.current_state.handle_response(self, user_text)

                # Étape 3 : Récupération du prompt de la phase actuelle
                base_sys_prompt = self.current_state.get_system_prompt()
                

                # Optionnel : Envoyer un trigger OSC pour animer l'avatar "en réflexion"
                self.network.send_osc("/etat", 2)

                logging.info("[Core] Checking the memory...")
                souvenirs = await self.memory.retrieve_context(user_text, top_k=2)

                if souvenirs:
                    sys_prompt = (
                        f"{base_sys_prompt}\n\n"
                        f"--- CONTEXTE / SOUVENIRS ---\n"
                        f"Voici des bribes de tes précédentes interactions avec l'utilisateur :\n{souvenirs}\n"
                        f"----------------------------\n"
                        f"Garde ces souvenirs en tête, mais n'y fais référence QUE si c'est pertinent "
                        f"pour répondre à la nouvelle requête de l'utilisateur."
                        f"Quand tu utilises ces souvenirs, tu peux les refomuler"
                    )
                    logging.info(f"[Core] {len(souvenirs.split('-')) - 1} parts of memory added to the prompt")
                else:
                    sys_prompt = base_sys_prompt
        

                # Étape 4 : Requête au modèle local (LM Studio)
                llm_response = await self.network.ask_llm(
                    system_prompt=sys_prompt, 
                    user_text=user_text
                )

                self.network.send_osc("/etat", 3)

                nouveau_souvenir = f"Utilisateur : {user_text}\nAIDAN : {llm_response}"
                # create_task lance l'enregistrement dans ChromaDB en parallèle. 
                asyncio.create_task(self.memory.add_memory(text_content=nouveau_souvenir, role="interaction"))

                # Étape 5 : Extraction flags, exécution des actions, et TTS
                await self.process_llm_response(llm_response)
                
                # Fin du cycle, on remet l'avatar en attente
                self.network.send_osc("/etat", 0)

            except asyncio.CancelledError:
                logging.info("[Core] Stopping audio loop")
                break
            except Exception as e:
                logging.error(f"[Core] Critical error in audio loop : {e}")
                await asyncio.sleep(1) # Évite un spam de la console en cas de crash
        
        
        async def force_interaction(self, forced_context: str):
            """
            Force l'IA à générer une réponse suite à un événement physique ou un ordre régie,
            """
            logging.info(f"[Core] ⚡ Événement injecté : {forced_context}")
            
            # 1. On vérifie si la phase doit changer (Ex: On bascule sur HotState)
            # Assure-toi d'importer HotState en haut de ton fichier !
            from states.hot_state import HotState
            await self.set_state(HotState())

            # 2. Récupération du prompt de ce nouvel état (qui est énervé)
            sys_prompt = self.current_state.get_system_prompt()

            self.network.send_osc("/etat", 2)

            # 3. On injecte l'événement comme si c'était un message utilisateur, 
            # mais avec une balise pour que l'IA comprenne le contexte.
            llm_response = await self.network.ask_llm(
                system_prompt=sys_prompt, 
                user_text=f"[ÉVÉNEMENT PHYSIQUE] {forced_context}"
            )

            self.network.send_osc("/etat", 3)

            # 4. Sauvegarde dans la mémoire ChromaDB
            nouveau_souvenir = f"Événement physique : {forced_context}\nRéaction d'AIDAN : {llm_response}"
            asyncio.create_task(self.memory.add_memory(text_content=nouveau_souvenir, role="system"))

            # 5. Extraction des flags et TTS (exactement comme le flux normal)
            await self.process_llm_response(llm_response)
            
            self.network.send_osc("/etat", 0)

    # --- Extrait de engine.py ---
async def process_user_input(self, user_text):
    # 1. Si on est dans le FinalState, on valide d'abord la réponse
    if isinstance(self.current_state, FinalState):
        await self.current_state.handle_response(self, user_text)
    
    # 2. On demande au LLM de générer sa réponse (qui contiendra la question suivante)
    # Le get_system_prompt() sera appelé et contiendra la nouvelle question indexée.
    response_text = await self.llm.generate(
        system_prompt=self.current_state.get_system_prompt(),
        user_input=user_text
    )
    
    # 3. Envoyer au TTS
    await self.audio.play_tts(response_text)