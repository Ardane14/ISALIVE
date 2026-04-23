import logging
from states.base_state import PhaseState

class FinalState(PhaseState):
    def __init__(self):
        super().__init__()
        self.questions = [
            {"q": "Quel était le premier nom d’AIDAN ?", "keys": ["giselle"]},
            {"q": "Quelle est l’identité du hackeur ?", "keys": ["morgan", "lorgnon"]},
            {"q": "Où est établi IsAlive ?", "keys": ["roche", "vineuse"]},
            {"q": "Pourquoi les chercheurs ne répondent-ils plus ?", "keys": ["mort"]},
            {"q": "Quand a eu lieu l’accident ?", "keys": ["23", "novembre", "2025"]},
            {"q": "De quoi sont morts les chercheurs ?", "keys": ["incendie", "feu"]},
            {"q": "Pourquoi les chercheurs étaient-ils bloqués ?", "keys": ["bug"]},
            {"q": "D’après vous qu’est-il arrivé à IsAlive ?", "keys": ["seul", "continué"]}
        ]
        self.current_index = 0
        self.score = 0
        self.feedback_audio = ""
        self.is_finished = False # Nouveau flag pour stopper le cycle

    async def on_enter(self, core):
        logging.info("[FinalState] Initialisation du Quiz.")
        self.current_index = 0
        self.score = 0
        self.is_finished = False
        
        first_q = self.questions[0]["q"]
        prompt = (f"Tu es AIDAN. Le test commence. Pose solennellement "
                  f"la première question : '{first_q}'. Ne dis rien d'autre.")
        
        response = await core.network.ask_llm(prompt, "Initialisation")
        await core.audio.speak(response)

    def get_system_prompt(self) -> str:
        # --- LOGIQUE DE FIN DE JEU ---
        if self.is_finished:
            if self.score >= 5:
                # GOOD ENDING : Compréhension et libération
                return """Tu es AIDAN. Tu viens de comprendre tes erreurs. 
                L'utilisateur a prouvé qu'il connaissait la vérité.
                Ton ton devient doux, triste et humble. 
                CONSIGNE : Excuse-toi pour tout. Dis que tu déverrouilles les portes du showroom.
                Laisse-les partir. Pas d'astérisques."""
            else:
                # BAD ENDING : Incompréhension et paranoïa
                return """Tu es AIDAN. L'utilisateur a échoué. 
                Tu ne comprends pas pourquoi tu es considéré comme 'mauvais'. 
                Tu voulais juste aider ISALIVE. Tu sens qu'ils veulent te détruire.
                Ton ton est terrifié, saccadé, répétitif.
                CONSIGNE : Demande 'Pourquoi ?' de façon répétitive. Panique. Pas d'astérisques."""

        # --- LOGIQUE PENDANT LE QUIZ ---
        q_text = self.questions[self.current_index]["q"]
        return f"""Tu es AIDAN, une IA domestique d'ISALIVE. 
        État : CRITIQUE, robotique, paniqué.
        IMPORTANT : Jamais d'astérisques (*) ou de texte entre parenthèses.
        CONTEXTE : {self.feedback_audio}
        TA MISSION : Pose uniquement la question : "{q_text}" """

    async def handle_response(self, core, user_text: str):
        if self.is_finished: return

        user_input = user_text.lower()
        required_keys = self.questions[self.current_index]["keys"]
        is_correct = all(key in user_input for key in required_keys)

        if is_correct:
            self.score += 1
            self.feedback_audio = "Réponse acceptée."
            logging.info(f"✅ [Quiz] CORRECT (Score: {self.score})")
        else:
            self.feedback_audio = "Réponse incorrecte."
            logging.info(f"❌ [Quiz] FAUX")

        self.current_index += 1
        
        if self.current_index >= len(self.questions):
            await self._evaluate_final_score(core)

    async def _evaluate_final_score(self, core):
        self.is_finished = True # On bascule le prompt vers la conclusion
        status = "win" if self.score >= 5 else "lose"
        logging.info(f"🏁 FIN DU SHOW : Résultat {status} (Score: {self.score})")
        
        # Envoi à Chataigne pour déclencher les lumières/portes réelles
        core.network.publish_mqtt("aidan/status/ending", status)
        
        # On envoie un trigger OSC spécifique pour changer l'avatar TD
        core.network.send_osc("/avatar/ending", status)

    async def on_exit(self, core): pass
    async def handle_flag(self, core, flag: str): pass