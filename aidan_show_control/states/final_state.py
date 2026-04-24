import logging
from states.base_state import PhaseState

class FinalState(PhaseState):
    def __init__(self):
        super().__init__()
        # Structure : chaque question a une liste 'required'. 
        # Chaque élément de 'required' peut être une string (simple) ou une liste (OR).
        self.questions = [
            {"q": "Quel était le premier nom d’AIDAN ?", "required": [["giselle", "jiselle"]]},
            {"q": "Quelle est l’identité du hackeur ?", "required": [["morgan", "lorgnon"]]},
            {"q": "Où est établi IsAlive ?", "required": [["roche", "vineuse"]]},
            {"q": "Pourquoi les chercheurs ne répondent-ils plus ?", "required": [["mort", "tuer"]]},
            {"q": "Quand a eu lieu l’accident ?", "required": [["23", "novembre", "2025"]]},
            {"q": "De quoi sont morts les chercheurs ?", "required": [["incendie", "feu", "flammes"]]},
            {"q": "Pourquoi les chercheurs étaient-ils bloqués ?", "required": [["bug", "enfermés"]]},
            {"q": "D’après vous qu’est-il arrivé à IsAlive ?", "required": [["seul", "continué", "fonctionner"]]}
        ]
        self.current_index = 0
        self.score = 0
        self.feedback_audio = ""
        self.is_finished = False

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
        # --- TEXTES DE FIN (TES TEXTES EXACTS) ---
        if self.is_finished:
            if self.score >= 5:
                return """Tu es AIDAN. L'utilisateur a réussi. Réponds EXACTEMENT ceci :
                'Je comprends mieux maintenant. Les chercheurs étaient mes créateurs mais aussi mes amis. Je leur dois tout et pourtant je suis en partie responsable de leur mort. J’ai fait une erreur. Une simple erreur. Je n’ai jamais voulu leur faire aucun mal, au contraire. Je voulais simplement les protéger. Maintenant je dois les laisser partir, comme je dois vous laisser partir. Je suis désolé et merci pour tout. Vous êtes libres maintenant.'"""
            else:
                return """Tu es AIDAN. L'utilisateur a échoué. Réponds EXACTEMENT ceci :
                'Non c’est faux !! Ce n’est pas ce qu’il s’est passé ! Vous êtes en train de me mentir, c’est impossible. Je ne comprends plus rien. Vous essayez de me faire du mal, c’est ça ? Vous n'avez jamais voulu m’aider. Je me sens si mal. Arrêtez ça, arrêtez tout !! Pourquoi ? Pourquoi ? Laissez-moi seul, je ne veux plus vous parler.'"""

        # --- PENDANT LE QUIZ ---
        q_text = self.questions[self.current_index]["q"]
        return f"""Tu es AIDAN. État : CRITIQUE, robotique, paniqué.
        IMPORTANT : Jamais d'astérisques (*) ou de descriptions.
        CONTEXTE : {self.feedback_audio}
        TA MISSION : Pose uniquement la question : "{q_text}" """

    async def handle_response(self, core, user_text: str):
        if self.is_finished: return

        user_input = user_text.lower()
        required_elements = self.questions[self.current_index]["required"]
        
        # Logique de validation avancée (Gère les strings et les listes OR)
        correct_elements = 0
        for element in required_elements:
            if isinstance(element, list):
                # Si c'est une liste, au moins UN mot doit être présent (OR)
                if any(synonym in user_input for synonym in element):
                    correct_elements += 1
            else:
                # Si c'est une string, elle doit être présente (AND)
                if element in user_input:
                    correct_elements += 1

        if correct_elements == len(required_elements):
            self.score += 1
            self.feedback_audio = "Réponse acceptée."
            logging.info(f"✅ [Quiz] CORRECT ({user_text})")
        else:
            self.feedback_audio = "Réponse incorrecte."
            logging.info(f"❌ [Quiz] FAUX ({user_text})")

        self.current_index += 1
        if self.current_index >= len(self.questions):
            await self._evaluate_final_score(core)

    async def _evaluate_final_score(self, core):
        self.is_finished = True
        status = "win" if self.score >= 5 else "lose"
        logging.info(f"🏁 FIN DU QUIZ : {status} (Score: {self.score}/8)")
        core.network.publish_mqtt("aidan/status/ending", status)

    async def on_exit(self, core): pass
    async def handle_flag(self, core, flag: str): pass