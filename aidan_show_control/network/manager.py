import asyncio
import aiohttp
import aiomqtt
from pythonosc import udp_client
import logging
import json

class NetworkManager:
    def __init__(self, network_config: dict, lm_config: dict):
        self.mqtt_broker = network_config.get('mqtt_broker', '192.168.0.10')
        self.mqtt_port = network_config.get('mqtt_port', 1883)
        self.osc_ip = network_config.get('osc_ip_av', '127.0.0.1')
        self.osc_port = network_config.get('osc_port_av', 8000)
        self.lm_url = lm_config.get('url', 'http://127.0.0.1:1234/v1/chat/completions')
        
        self.lm_health_url = self.lm_url.replace("/chat/completions", "/models")
        self.lm_timeout = lm_config.get('timeout_seconds', 5.0)

        self.http_session = None
        self.mqtt_client = None 
        self.osc_client = udp_client.SimpleUDPClient(self.osc_ip, self.osc_port)
        self.lm_is_online = False

    async def connect_http(self):
        self.http_session = aiohttp.ClientSession()
        logging.info("[Network] Session HTTP ouverte.")

    async def close_all(self):
        if self.http_session:
            await self.http_session.close()

    def send_osc(self, address: str, value):
        try:
            self.osc_client.send_message(address, value)
        except Exception as e:
            logging.error(f"[OSC] Erreur: {e}")

    async def watchdog_lmstudio(self):
        while True:
            try:
                async with self.http_session.get(self.lm_health_url, timeout=2.0) as response:
                    if response.status == 200:
                        if not self.lm_is_online:
                            logging.info("[Watchdog] LM Studio Online")
                            self.lm_is_online = True
                    else: raise Exception()
            except Exception:
                if self.lm_is_online:
                    logging.error("[Watchdog] LM Studio Offline")
                    self.lm_is_online = False
            await asyncio.sleep(5)

    async def listen_mqtt(self, on_message_callback):
        reconnect_interval = 2
        while True:
            try:
                async with aiomqtt.Client(self.mqtt_broker, port=self.mqtt_port) as client:
                    logging.info("[MQTT] Connecté au Broker")
                    self.mqtt_client = client 
                    await client.subscribe("aidan/control/#")
                    async for message in client.messages:
                        topic = message.topic.value
                        payload = message.payload.decode('utf-8')
                        await on_message_callback(topic, payload)
            except aiomqtt.MqttError:
                self.mqtt_client = None
                await asyncio.sleep(reconnect_interval)

    # --- MÉTHODE BIEN INDENTÉE ---
    async def publish_mqtt(self, topic: str, payload):
        """Publie un message. Gère dict, list et string automatiquement."""
        if self.mqtt_client is not None:
            try:
                if isinstance(payload, (dict, list)):
                    payload = json.dumps(payload)
                elif not isinstance(payload, str):
                    payload = str(payload)
                
                await self.mqtt_client.publish(topic, payload=payload, retain=True)
                logging.info(f"[MQTT OUT] {topic} : {payload}")
            except Exception as e:
                logging.error(f"[MQTT] Erreur publication: {e}")
        else:
            logging.warning(f"[MQTT] Non connecté, échec sur : {topic}")

    async def ask_llm(self, system_prompt: str, user_text: str) -> str:
        if not self.http_session: return ""
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "max_tokens": 2048
        }
        try:
            timeout = aiohttp.ClientTimeout(total=self.lm_timeout)
            async with self.http_session.post(self.lm_url, json=payload, timeout=timeout) as resp:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            return "Désolé, j'ai eu un petit problème de connexion."