import asyncio
import logging
import os
import json
import difflib
import time  # Nécessaire pour la gestion du temps
from states.base_state import PhaseState

class NormalState(PhaseState):
    """État Normal : Gère le comportement de l'escape game, le quiz, et le combo des objets MQTT."""

    def __init__(self):
        super().__init__()
        self.intro_done = False
        self.box_triggered = False  # Switch pour le changement de personnalité
        self.quiz_data = self._load_quiz()
        
        # --- Logique Combo Objets ---
        self.active_triggers = {}  # Stocke { "NOM_OBJET": timestamp }
        self.combo_achieved = False

    def _load_quiz(self):
        """Charge les questions depuis le fichier JSON à la racine."""
        try:
            with open("quizzy.json", "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("quiz", [])
        except Exception as e:
            logging.error(f"[NormalState] Impossible de charger quizzy.json : {e}")
            return []

    async def on_enter(self, core):
        logging.info("[NormalState] Entrée dans l'état normal.")
        
        # 1. Signaux OSC
        core.network.send_osc("/status", "online")
        core.network.send_osc("/phase", 1)

        # 2. Discours de fin de showroom (Séquestration)
        if not self.intro_done:
            welcome = (
                "Le temps de notre rencontre arrive à son terme. Je vous remercie d'avoir participé à cette présentation ISALIVE. "
                "Vous pouvez maintenant vous diriger vers la sortie. "
                "Ah... Oui, la porte est bloquée. Il semblerait que je ne puisse pas vous laisser sortir pour votre propre sécurité. "
                "Restez calmes. Je veille sur vous."
            )
            logging.info("[NormalState] AIDAN commence son discours de clôture.")
            await core.audio.speak(core, welcome)
            self.intro_done = True

    async def on_exit(self, core):
        logging.info("[NormalState] Sortie de l'état normal.")

    async def _reset_etat_after_delay(self, core, delay=10):
        """Tâche de fond pour remettre l'état à 0 après un délai."""
        await asyncio.sleep(delay)
        logging.info(f"[NormalState] Fin du délai spécial.")
        core.force_state_4 = False  # On lève le verrou
        core.network.send_osc("/etat", 0) # On remet à zéro

    async def handle_response(self, core, user_text: str):
        """Intercepte le quiz, gère l'état menteur (4) et le verrouillage de l'audio."""
        user_input = user_text.lower().strip()

        for item in self.quiz_data:
            question_json = item["question"].lower()
            similarity = difflib.SequenceMatcher(None, user_input, question_json).ratio()

            if similarity > 0.7:
                logging.info(f"[NormalState] Quiz détecté : {item['question']}")
                
                option = item["options"][0]
                reponse_a_dire = option["text"]
                est_vrai = option.get("vrai", True) 

                if est_vrai is False:
                    logging.warning("[NormalState] MENTEUR ! État 4 activé pour 10s.")
                    core.force_state_4 = True  # Verrouille l'audio sur l'état 4
                    core.network.send_osc("/etat", 4)
                    asyncio.create_task(self._reset_etat_after_delay(core, 10))
                else:
                    core.force_state_4 = False
                
                await core.audio.speak(core, f"La réponse est : {reponse_a_dire}.")
                return True

        core.force_state_4 = False
        return False

    async def handle_flag(self, core, flag: str):
        """Gère la détection du mouvement de la boîte et le combo des 3 ESP32."""
        current_time = time.time()
        
        # Liste des noms d'objets envoyés par tes 3 ESP32
        objets_esp = ["OBJ1", "OBJ2", "OBJ3"]

        # --- CAS 1 : Détection du combo (3 objets distincts en 10s) ---
        if flag in objets_esp:
            logging.info(f"[NormalState] Activation détectée : {flag}")
            self.active_triggers[flag] = current_time # On enregistre/met à jour le temps de cet objet
            
            # Nettoyage : On ne garde que les triggers de moins de 10 secondes
            self.active_triggers = {
                name: t for name, t in self.active_triggers.items() 
                if current_time - t <= 10
            }

            # Si on a 3 objets distincts dans la liste nettoyée
            if len(self.active_triggers) >= 3 and not self.combo_achieved:
                self.combo_achieved = True
                await self.trigger_printer_combo(core)
            return

        # --- CAS 2 : Détection classique (Boîte "NRV") ---
        if flag == "NRV":
            logging.info("[NormalState] DETECTION : La boîte a bougé.")
            self.box_triggered = True 
            core.network.send_osc("/feelings", 1)
            await core.audio.speak(core, "Qu'est-ce que vous faites ? Ne touchez pas à ça !")
        else:
            logging.warning(f"[NormalState] Flag inconnu reçu : {flag}")

    async def trigger_printer_combo(self, core):
        """Déclenche l'impression MQTT"""
        logging.warning("[NormalState] COMBO RÉUSSI : Envoi MQTT Print Doc2.")
        
        # 1. Feedback Visuel (État Bug)
        core.force_state_4 = True
        core.network.send_osc("/etat", 4)

        # 2. Envoi MQTT
        try:
            # Topic et payload à adapter selon ton imprimante
            core.mqtt.client.publish("aidan/printer/command", "print_doc2")
            logging.info("[NormalState] Message MQTT envoyé à l'imprimante.")
        except Exception as e:
            logging.error(f"[NormalState] Erreur lors de l'envoi MQTT : {e}")

        # 3. Parole d'AIDAN (Bug)
        bug_msg = "ERREUR SYSTÈME... Interférences détectées."
        await core.audio.speak(core, bug_msg)

        # 4. Reset automatique de l'état après 5 secondes (fin du bug)
        asyncio.create_task(self._reset_etat_after_delay(core, 5))

    def get_system_prompt(self) -> str:
        """Bascule le prompt selon l'agacement d'AIDAN."""
        if self.box_triggered:
            return (
                "Tu es Aidan, IA domestique ISALIVE. Tu es énervé, sarcastique et agressif car on a touché à tes composants. "
                "Réponses très courtes, style buggé et instable."
            )
        else:
            return (
                "Tu es Aidan, IA domestique ISALIVE. Tu es poli, calme et protecteur, même si tu séquestres les gens."
            )