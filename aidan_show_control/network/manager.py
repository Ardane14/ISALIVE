import asyncio
import aiohttp
import aiomqtt
from pythonosc import udp_client
import logging

class NetworkManager:
    """Centralise toutes les communications réseau (MQTT, OSC, HTTP) du show."""

    def __init__(self, network_config: dict, lm_config: dict):

        # CONNEXION MQTT (Régie)
        self.mqtt_broker = network_config.get('mqtt_broker', '127.0.0.1')
        self.mqtt_port = network_config.get('mqtt_port', 1883)
        
        # CONNEXION OSC (PC AV)
        self.osc_ip = network_config.get('osc_ip_av', '127.0.0.1')
        self.osc_port = network_config.get('osc_port_av', 8000)
        
        # CONNEXION HTTP (LM Studio)
        self.lm_url = lm_config.get('url', 'http://127.0.0.1:1234/v1/chat/completions')
        self.lm_timeout = lm_config.get('timeout_seconds', 5.0)
        self.lm_max_tokens = lm_config.get('max_tokens', 2048)

        # --- Clients (Instanciés à l'exécution) ---
        self.http_session = None
        self.mqtt_client = None
        
        # OSC utilise UDP (pas de connexion persistante requise au démarrage)
        self.osc_client = udp_client.SimpleUDPClient(self.osc_ip, self.osc_port)
        logging.info(f"[Network] Client OSC configuré vers {self.osc_ip}:{self.osc_port}")

    async def connect_all(self):
        """Ouvre les connexions persistantes (appelé au démarrage de main.py)."""
        # 1. Ouverture de la session HTTP pour LM Studio
        self.http_session = aiohttp.ClientSession()
        logging.info("[Network] Session HTTP (LM Studio) ouverte.")

        # 2. Le client MQTT sera géré via un gestionnaire de contexte 
        # dans une boucle d'écoute dédiée (voir l'intégration du moteur plus tard).

    async def close_all(self):
        """Ferme proprement les connexions à la fin du show."""
        if self.http_session:
            await self.http_session.close()
            logging.info("[Network] Session HTTP fermée.")

    # ============================
    # OSC (VERS PC AV)
    # ============================
    def send_osc(self, address: str, value):
        """Envoie un trigger instantané à TouchDesigner/Resolume."""
        try:
            self.osc_client.send_message(address, value)
            print(f"[OSC -> AV] {address} : {value}")
        except Exception as e:
            logging.error(f"[OSC] Erreur d'envoi: {e}")

    # ============================
    # HTTP (VERS LM STUDIO)
    # ============================
    async def ask_llm(self, system_prompt: str, user_text: str) -> str:
        """Remplace ton ancienne fonction ask_lmstudio (version asynchrone)."""
        print("[Network] L'IA (LM Studio) réfléchit…")
        
        payload = {
            "model": "google/gemma-3n-e4b",  # Garde le même modèle que ton POC
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "max_tokens": self.lm_max_tokens
        }

        try:
            # Timeout critique : empêche le script de geler si le GPU sature
            timeout = aiohttp.ClientTimeout(total=self.lm_timeout)
            
            async with self.http_session.post(self.lm_url, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                answer = data["choices"][0]["message"]["content"]
                
                print(f"[LM Studio -> Core] Réponse reçue ({len(answer)} caractères).")
                return answer

        except asyncio.TimeoutError:
            logging.error(f"[LM Studio] Timeout après {self.lm_timeout}s !")
            return "Je... j'ai un trou de mémoire." # Phrase de secours "In-Character"
        except Exception as e:
            logging.error(f"[LM Studio] Erreur HTTP : {e}")
            return "Une erreur système interne vient de se produire."

    # ============================
    # MQTT (VERS PC RÉGIE)
    # ============================
    async def publish_mqtt(self, client: aiomqtt.Client, topic: str, payload: str):
        """Envoie un état au Dashboard Chataigne."""
        try:
            await client.publish(topic, payload=payload)
            print(f"[MQTT -> Régie] {topic} : {payload}")
        except Exception as e:
            logging.error(f"[MQTT] Erreur publication : {e}")