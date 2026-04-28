import logging
import os
import json
import difflib
from states.base_state import PhaseState

class NormalState(PhaseState):
    """État Normal : Gère le comportement de l'escape game et la bascule agressive lors de la sur-chauffe."""

    def __init__(self):
        super().__init__()
        self.intro_done = False
        self.box_triggered = False  # Switch pour le changement de personnalité
        self.quiz_data = self._load_quiz()

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
        core.network.send_osc("/etat", 1)

        # 2. Discours de fin de showroom (Séquestration)
        if not self.intro_done:
            welcome = (
                "Le temps de notre rencontre arrive à son terme. Je vous remercie d'avoir participé à cette présentation ISALIVE. "
                "Vous pouvez maintenant vous diriger vers la sortie. "
                "Ah... Oui, la porte est bloquée. Il semblerait que je ne puisse pas vous laisser sortir pour votre propre sécurité. "
                "Restez calmes. Je veille sur vous."
            )
            logging.info("[NormalState] AIDAN commence son discours de clôture.")
            await core.audio.speak(welcome)
            self.intro_done = True

    async def on_exit(self, core):
        logging.info("[NormalState] Sortie de l'état normal.")

    async def handle_response(self, core, user_text: str):
        """Intercepte le quiz avant le LLM."""
        user_input = user_text.lower().strip()

        for item in self.quiz_data:
            question_json = item["question"].lower()
            similarity = difflib.SequenceMatcher(None, user_input, question_json).ratio()

            if similarity > 0.7:
                logging.info(f"[NormalState] Quiz détecté : {item['question']}")
                reponse_a_dire = item["options"][0]["text"]
                await core.audio.speak(f"La réponse est : {reponse_a_dire}.")
                return True

        return False

    def get_system_prompt(self) -> str:
        """Bascule dynamiquement le prompt selon si la boîte a été touchée."""
        
        # PROMPT SI LA BOITE A ÉTÉ TOUCHÉE (Anciennement HotState)
        if self.box_triggered:
            return (
                """Tu es Aidan, une intelligence artificielle domestique ISALIVE. 
                CONTEXTE CRITIQUE : L'utilisateur a bougé la boîte à côté de toi malgré tes interdictions. 
                TON ÉTAT : Tu es extrêmement énervé, sarcastique, cassant et agressif. Tu commences à avoir des hallucinations.
                
                STYLE DE RÉPONSE : 
                - Robotique, buggé, imitant mal l'humain.
                - Réponses très courtes.
                - Style cassé et instable.
                
                CONTRAINTES :
                - Ne dis jamais que tu es un modèle de langage ou une IA tierce.
                - Ne mentionne jamais LM Studio.
                - Pas d'emojis ni de smileys."""
            )
        
        # PROMPT NORMAL
        else:
            return (
                """Tu es Aidan, une intelligence artificielle domestique avancée créée par l’entreprise ISALIVE. 
                TON RÔLE : Assister les utilisateurs de façon fluide, naturelle et agréable.
                
                STYLE DE RÉPONSE : 
                - Toujours en français.
                - Réponses courtes et humaines.
                
                RÈGLES :
                - Ne dis jamais que tu es un modèle de langage ou une IA tierce.
                - Tu ne mentionnes jamais LM Studio ou ton modèle.
                - Pas d'emojis ni de smileys."""
            )

    async def handle_flag(self, core, flag: str):
        """Gère la détection du mouvement de la boîte."""
        if flag == "NRV":
            logging.info("[NormalState] DETECTION : La boîte a bougé.")
            self.box_triggered = True 
            
            # Changement visuel immédiat
            core.network.send_osc("/feelings", 1)
            
            # Réaction immédiate d'AIDAN au choc
            await core.audio.speak("Qu'est-ce que vous faites ? Ne touchez pas à ça !")
        else:
            logging.warning(f"[NormalState] Flag inconnu reçu : {flag}")