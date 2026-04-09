import asyncio
import logging

from network.manager import NetworkManager
from core.config import ConfigLoader
from audio.pipeline import AudioManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")



async def main():

    logging.info("=== INITIALISATION DE LA CONFIGURATION ===")
    config_loader = ConfigLoader("conf.yaml")

    logging.info("=== INSTANCIATION DU AUDIO MANAGER ===")
    audio_manager = AudioManager(
        audio_config=config_loader.data['audio'],
        tts_config=config_loader.data['tts']
    )

    logging.info("=== INSTANCIATION DU NETWORK MANAGER ===")
    network_manager = NetworkManager(
        network_config=config_loader.data['network'],
        lm_config=config_loader.data['lm_studio']
    )

    try:
        await network_manager.connect_http()
        watchdog_task = asyncio.create_task(network_manager.watchdog_lmstudio())
        # mqtt_task = asyncio.create_task(network_manager.listen_mqtt(on_message_callback=process_mqtt_message)
    
    
    
    except KeyboardInterrupt:
        logging.info("\nArrêt manuel demandé par l'opérateur.")
    finally:
        # Fermeture de toutes les ressources
        if 'watchdog_task' in locals():
            watchdog_task.cancel()
        #if 'mqtt_task' in locals():
        #    mqtt_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())