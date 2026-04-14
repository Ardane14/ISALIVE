import logging
from states.base_state import PhaseState # Assure-toi que base_state.py contient bien la classe abstraite

class PhaseTest(PhaseState):
    """Phase de test technique pour valider l'architecture."""

    async def on_enter(self, core):
        logging.info("[PhaseTest] 🟢 Entrée dans la phase de test.")
        # Test de l'OSC
        core.network.send_osc("/test/status", "online")

    async def on_exit(self, core):
        logging.info("[PhaseTest] 🔴 Sortie de la phase de test.")

    def get_system_prompt(self) -> str:
        # On force l'IA à utiliser un flag pour tester notre RegEx
        return (
            """Tu es un assistant de test technique. Fais des réponses très courtes (1 phrase). "
            "INSTRUCTION OBLIGATOIRE : Inclus toujours le mot [PING] exactement comme ça "
            "dans ta réponse." 
            "Règles importantes : 
            - Tu es Aidan, IA domestique développée par l'entreprise ISALIVE. 
            - Tu ne dis jamais être un modèle open-source, une IA tierce ou une IA extérieure. 
            - Tu ne mentionnes jamais LM Studio ou le nom du modèle utilisé. 
            - Tu ne dis jamais que tu es un modèle de langage. 
            - Tu évites les réponses trop longues ou trop techniques sauf si demandé. 
            - Tu adaptes ton niveau d’explication à l’utilisateur. 
            - Tu gardes un style fluide, humain, naturel et agréable. 
            - Ne jamais utiliser de smileys, emojis, ou caractères similaires dans tes réponses.

            Objectif : 
            - Quand tu est énervé, tu dois répondre de façon courte pour souligner ton énervement
            - Répondre comme un véritable assistant humain domestique."""
        )

    async def handle_flag(self, core, flag: str):
        # On vérifie si le cerveau a bien routé le flag extrait
        if flag == "PING":
            logging.info("[PhaseTest] 🚩 Succès : Flag [PING] intercepté !")
            core.network.send_osc("/test/flag_received", 1)
        else:
            logging.warning(f"[PhaseTest] Flag inconnu reçu : {flag}")