import asyncio
import logging
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from core.config import ConfigLoader
from audio.pipeline import AudioManager
from network.manager import NetworkManager
from core.engine import AidanCore
from states.phase_test import PhaseTest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

async def main():
    logging.info("=== INITIALISATION DE AIDAN ===")
    config_loader = ConfigLoader("conf.yaml")

    audio_manager = AudioManager(
        audio_config=config_loader.data.get('audio', {}),
        tts_config=config_loader.data.get('tts', {})
    )
    
    network_manager = NetworkManager(
        network_config=config_loader.data.get('network', {}),
        lm_config=config_loader.data.get('lm_studio', {})
    )

    # 1. Instanciation du Cerveau
    aidan_core = AidanCore(config_loader.data, network_manager, audio_manager)

    try:
        await network_manager.connect_http()

        # 2. On passe la fonction 'handle_mqtt_message' du cerveau comme Callback !
        # Ainsi, quand la régie parle, le cerveau écoute directement.
        watchdog_task = asyncio.create_task(network_manager.watchdog_lmstudio())
        mqtt_task = asyncio.create_task(network_manager.listen_mqtt(on_message_callback=aidan_core.handle_mqtt_message))

        logging.info("=== MOTEUR EN LIGNE ===")
        
        # 3. On force l'état initial pour notre test
        await aidan_core.set_state(PhaseTest())
        
        # 4. On lance la boucle infinie de l'IA (qui remplace notre ancien 'while True')
        await aidan_core.run_audio_loop()

    except KeyboardInterrupt:
        logging.info("\nArrêt manuel demandé.")
    finally:
        # Fermeture propre
        if 'watchdog_task' in locals(): watchdog_task.cancel()
        if 'mqtt_task' in locals(): mqtt_task.cancel()
        await network_manager.close_all()
        logging.info("=== EXTINCTION PROPRE ===")

if __name__ == "__main__":
    asyncio.run(main())