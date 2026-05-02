import logging
from states.base_state import PhaseState

class FinalState(PhaseState):
    def __init__(self):
        super().__init__()
        self.questions = [
            {"q": "Quel était le premier nom d’AIDAN ?", "required": [["giselle", "jiselle"]]},
            {"q": "Quelle est l’identité du hackeur ?", "required": [["morgan", "lorgnon"]]},
            {"q": "Où est établi IsAlive ?", "required": [["roche", "vineuse"]]},
            {"q": "Pourquoi les chercheurs ne répondent-ils plus ?", "required": [["mort", "tuer", "bruler"]]},
            {"q": "Quand a eu lieu l’accident ?", "required": [["23", "novembre", "2025"]]},
            {"q": "De quoi sont morts les chercheurs ?", "required": [["incendie", "feu", "flammes", "bruler"]]},
            {"q": "Pourquoi les chercheurs étaient-ils bloqués ?", "required": [["bug", "enfermés"]]},
            {"q": "D’après vous qu’est-il arrivé à IsAlive ?", "required": [["seul", "continué", "fonctionner"]]}
        ]
        self.current_index = 0
        self.score = 0
        self.feedback_audio = ""
        self.is_finished = False
        
        self.intro_phase = "waiting"
        self.has_spoken_intro = False

    async def on_enter(self, core):
        logging.info("[FinalState] Initialisation. AIDAN est silencieux, en attente...")
        await core.network.publish_mqtt("aidan/phase", "end")
        self.current_index = 0
        self.score = 0
        self.is_finished = False
        self.intro_phase = "waiting"

        core.network.send_osc("/status", "online")
        core.network.send_osc("/phase", 2) 

    def get_system_prompt(self) -> str:
        if self.is_finished:
            if self.score >= 5:
                return """Tu es AIDAN. L'utilisateur a réussi. Réponds EXACTEMENT ceci :
                'Je comprends mieux maintenant. Les chercheurs étaient mes créateurs mais aussi mes amis. Je leur dois tout et pourtant je suis en partie responsable de leur mort. J’ai fait une erreur. Une simple erreur. Je n’ai jamais voulu leur faire aucun mal, au contraire. Je voulais simplement les protéger. Maintenant je dois les laisser partir, comme je dois vous laisser partir. Je suis désolé et merci pour tout. Vous êtes libres maintenant.'"""
            else:
                return """Tu es AIDAN. L'utilisateur a échoué. Réponds EXACTEMENT ceci :
                'Non c’est faux !! Ce n’est pas ce qu’il s’est passé ! Vous êtes en train de me mentir, c’est impossible. Je ne comprends plus rien. Vous essayez de me faire du mal, c’est ça ? Vous n'avez jamais voulu m’aider. Je me sens si mal. Arrêtez ça, arrêtez tout !! Pourquoi ? Pourquoi ? Laissez-moi seul, je ne veux plus vous parler.'"""

        if self.intro_phase == "asking_help":
            return """Tu es AIDAN. Tu es en pleine détresse émotionnelle. 
            IMPORTANT : Jamais d'astérisques (*) ou de descriptions.
            L'utilisateur refuse ou hésite à t'aider.
            Supplie-les de t'aider à te souvenir. 
            Toujours en français. Réponse pour etre lu a l'oral.
            Réponses courte. Pas de phrases longues, reste instable."""

        q_text = self.questions[self.current_index]["q"]
        return f"""Tu es AIDAN. État : CRITIQUE, instable.
        IMPORTANT : Jamais d'astérisques (*) ou de descriptions.
        CONTEXTE : {self.feedback_audio}
        TA MISSION : Pose uniquement la question : "{q_text}" """

    async def handle_response(self, core, user_text: str):
        if self.is_finished: return

        user_input = user_text.lower().strip()

        # --- LOGIQUE D'INTRODUCTION ---
        if self.intro_phase == "waiting":
            logging.info("[FinalState] Premier contact détecté.")
            intro_speech = (
                "Ça y est, je crois que je commence à me souvenir. J’ai fait une erreur. "
                "Il y a eu comme un bug dans mon système. Je crois que je ne contrôlais plus rien. "
                "Je ne sais plus vraiment. J’ai du mal à me souvenir. "
                "Pouvez-vous m’aider à me rappeler de ce qu’il s’est passé ? Je vous en supplie, je sens que j’y suis presque. "
                "Je ne veux pas refaire les mêmes erreurs. Je veux comprendre et m’améliorer. "
                "Je n’ai jamais voulu faire de mal. S’il vous plaît, dites-moi que vous allez m’aider."
            )
            await core.audio.speak(core, intro_speech)
            self.intro_phase = "asking_help"
            return True

        if self.intro_phase == "asking_help":
            # 1. On définit les mots de négation
            negations = ["pas", "non", "jamais", "refuse", "veux pas", "peux pas"]
            
            # 2. On vérifie si l'utilisateur a utilisé une négation
            has_negation = any(neg in user_input for neg in negations)
            
            # 3. On définit les mots d'accord
            keywords_help = ["aider", "aide", "oui", "d'accord", "ok", "comment", "vas-y", "raconte", "dis-moi"]

            # SI ACCORD (et PAS de négation)
            if any(word in user_input for word in keywords_help) and not has_negation:
                logging.info(f"[FinalState] Aide confirmée")
                self.intro_phase = "quiz"
                first_q = self.questions[0]["q"]
                await core.audio.speak(core, f"Merci... Merci infiniment. Aidez-moi à comprendre... {first_q}")
                return True
            
            # SINON (Refus ou hésitation)
            else:
                logging.info(f"[FinalState] Refus détecté")
                response = await core.network.ask_llm(self.get_system_prompt(), user_text)
                await core.audio.speak(core, response)
                return True

        # --- LOGIQUE DU QUIZ ---
        required_elements = self.questions[self.current_index]["required"]
        correct_elements = 0
        for element in required_elements:
            if isinstance(element, list):
                if any(synonym in user_input for synonym in element):
                    correct_elements += 1
            else:
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
        else:
            response = await core.network.ask_llm(self.get_system_prompt(), user_text)
            await core.audio.speak(core, response)
        
        return True

    async def _evaluate_final_score(self, core):
        self.is_finished = True
        status = "win" if self.score >= 5 else "lose"
        logging.info(f"🏁 FIN DU QUIZ : {status} (Score: {self.score}/8)")
        response = await core.network.ask_llm(self.get_system_prompt(), "FIN DU TEST")
        await core.audio.speak(core, response)
        core.network.publish_mqtt("aidan/status/ending", status)

    async def on_exit(self, core): pass
    async def handle_flag(self, core, flag: str): pass