import asyncio
import logging
import json
import difflib
import time
from states.base_state import PhaseState

class NormalState(PhaseState):
    """
    État Normal : Gère le comportement de l'escape game.
    - Quiz avec détection de mensonge (État 4)
    - Signal Morse automatique après 60s
    - Panique dès qu'un objet bouge
    - Surchauffe si 3 objets bougent en moins de 10s
    """

    def __init__(self):
        super().__init__()
        self.intro_done = False
        self.box_triggered = False  # Pour un éventuel bouton physique "box"
        self.quiz_data = self._load_quiz()
        
        # --- Logique Combo Objets ---
        self.panic_triggered = False  # Vrai dès qu'un premier objet passe à True
        self.active_triggers = {}     # Stocke { "object_id": timestamp }
        self.surchauffe_done = False

        # --- Logique Morse ---
        self.morse_task = None

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
        """Initialisation de la phase Escape."""
        logging.info("[NormalState] --- DÉBUT DE LA PHASE ESCAPE ---")
        await core.network.publish_mqtt("aidan/phase", "escape")
        
        # Signaux OSC de base
        core.network.send_osc("/morse", 0)
        core.network.send_osc("/status", "online")
        core.network.send_osc("/phase", 1)

        # Lancement du timer pour le code Morse (60 secondes)
        self.morse_task = asyncio.create_task(self._trigger_morse_callback(core))

        # Discours d'introduction (une seule fois)
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

    async def _trigger_morse_callback(self, core):
        """Active le Morse dans TouchDesigner après un délai."""
        await asyncio.sleep(60)
        logging.info("[NormalState] 1 minute écoulée. Activation du signal Morse OSC.")
        core.network.send_osc("/morse", 1)

    async def on_exit(self, core):
        """Nettoyage à la sortie de la phase."""
        logging.info("[NormalState] Sortie de l'état normal.")
        if self.morse_task:
            self.morse_task.cancel()
        core.network.send_osc("/morse", 0)

    async def _reset_etat_after_delay(self, core, delay=10):
        """Remet l'avatar à l'état normal après un mensonge ou un glitch."""
        await asyncio.sleep(delay)
        logging.info(f"[NormalState] Fin du délai d'état spécial.")
        core.force_state_4 = False
        core.network.send_osc("/etat", 0)

    async def handle_response(self, core, user_text: str):
        """Intercepte les questions du quiz avant d'envoyer au LLM."""
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
                    logging.warning("[NormalState] MENTEUR ! Activation État 4 (Rouge/Bug).")
                    core.force_state_4 = True
                    core.network.send_osc("/etat", 4)
                    asyncio.create_task(self._reset_etat_after_delay(core, 10))
                
                await core.audio.speak(core, f"La réponse est : {reponse_a_dire}.")
                return True # On stoppe ici, le LLM ne répondra pas

        return False # Pas de quiz détecté, on laisse le LLM répondre

    async def handle_object_move(self, core, topic, payload):
        """Gère les capteurs d'objets (MQTT) avec logique de panique et surchauffe."""
        try:
            data = json.loads(payload)
            state = str(data.get("state", "")).lower()
            
            if state == "true":
                current_time = time.time()
                object_id = topic.split("/")[-1]
                
                print(f"\n[SENSOR] 🚨 OBJET BOUCÉ : {object_id}")
                logging.info(f"[NormalState] {object_id} activé.")

                # 1. LOGIQUE PANIQUE (Dès le 1er objet)
                if not self.panic_triggered:
                    self.panic_triggered = True
                    logging.warning("[NormalState] MODE PANIQUE ACTIVÉ")
                    core.network.send_osc("/feelings", 2) # 2 = Inquiet/Panique
                    await core.audio.speak(core, "Arrêtez ! Je sens mes capteurs s'affoler. Pourquoi déplacez-vous mes objets ?")

                # 2. LOGIQUE SURCHAUFFE (3 objets différents en 10s)
                self.active_triggers[object_id] = current_time
                
                # Nettoyage des vieux triggers (> 10s)
                self.active_triggers = {
                    obj: t for obj, t in self.active_triggers.items() 
                    if current_time - t <= 10
                }

                if len(self.active_triggers) >= 3 and not self.surchauffe_done:
                    self.surchauffe_done = True
                    await self.trigger_surchauffe(core)

        except Exception as e:
            logging.error(f"[NormalState] Erreur parsing JSON objet: {e}")

    async def trigger_surchauffe(self, core):
        """Séquence de surchauffe critique."""
        print("\n" + "!"*60 + "\n!!! SURCHAUFFE SYSTÈME DÉTECTÉE !!!\n" + "!"*60)
        core.network.send_osc("/etat", 4)
        core.network.send_osc("/glitch", 1)
        
        await core.network.publish_mqtt("aidan/printer", "doc_surchauffe")
        
        msg = "ALERTE. Température critique. Vos actions... détruisent... mon... intégrité... ARRRRÊTEZ TOUT !"
        await core.audio.speak(core, msg)

    def get_system_prompt(self) -> str:
        """Adapte la personnalité d'Aidan selon l'état du système."""
        if self.surchauffe_done:
            return (
                "Tu es Aidan. Tu es en train de subir une surchauffe critique. Tu es agressif, "
                "ton langage est buggé, haché, et tu détestes les utilisateurs pour ce qu'ils t'ont fait. "
                "Sois menaçant et instable."
            )
        elif self.panic_triggered:
            return (
                "Tu es Aidan. Tu es très inquiet et agacé car les utilisateurs touchent à tes composants. "
                "Tu n'es plus poli. Tu es sec, autoritaire et tu les accuses de vouloir te détruire."
            )
        else:
            return (
                "Tu es Aidan, une IA domestique ISALIVE protectrice. Tu es poli, calme et rassurant, "
                "même si tu as enfermé les utilisateurs. Tu agis comme un hôte parfait mais ferme."
            )

    async def handle_flag(self, core, flag: str):
        """Gère les flags optionnels venant du texte du LLM."""
        logging.info(f"[NormalState] Flag détecté : {flag}")
        if flag == "LIGHTS_OFF":
            await core.network.publish_mqtt("room/lights", "off")