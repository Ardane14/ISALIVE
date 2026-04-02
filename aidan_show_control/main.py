import asyncio
from network.manager import NetworkManager
from core.config import ConfigLoader
from audio.pipeline import AudioManager

async def main():
    # 1. Chargement du fichier YAML
    config_loader = ConfigLoader("conf.yaml")

    # 2. Instanciation de l'AudioManager (Injection des dépendances)
    audio_manager = AudioManager(
        audio_config=config_loader.data['audio'],
        tts_config=config_loader.data['tts']
    )

    network_manager = NetworkManager(
        network_config=config_loader.data['network'],
        lm_config=config_loader.data['lm_studio']
    )

    # --- Exemple d'utilisation dans ta boucle ---
    # Fichier_audio = await audio_manager.record_until_silence()
    # Texte_transcrit = await audio_manager.transcribe(Fichier_audio)
    # await audio_manager.speak("Bonjour, l'architecture fonctionne.")

if __name__ == "__main__":
    asyncio.run(main())