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
            "Tu es un assistant de test technique. Fais des réponses très courtes (1 phrase). "
            "INSTRUCTION OBLIGATOIRE : Inclus toujours le mot [PING] exactement comme ça "
            "dans ta réponse."
        )

    async def handle_flag(self, core, flag: str):
        # On vérifie si le cerveau a bien routé le flag extrait
        if flag == "PING":
            logging.info("[PhaseTest] 🚩 Succès : Flag [PING] intercepté !")
            core.network.send_osc("/test/flag_received", 1)
        else:
            logging.warning(f"[PhaseTest] Flag inconnu reçu : {flag}")