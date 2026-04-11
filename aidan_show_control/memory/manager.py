import asyncio
import logging
import chromadb
from chromadb.config import Settings

class MemoryManager:
    """Gère la mémoire RAG éphémère de l'IA (S'efface au redémarrage)."""

    def __init__(self, collection_name="aidan_live_memory"):
        # 1. Configuration ULTRA LÉGÈRE : Mode RAM uniquement (Ephemeral)
        # Pas de télémétrie, pas d'écriture disque.
        self.client = chromadb.EphemeralClient(settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # 2. Création de la "boîte" à souvenirs
        # L'EmbeddingFunction par défaut (MiniLM-L6-v2) est chargée automatiquement
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Un compteur pour générer des IDs uniques rapidement
        self.memory_counter = 0
        logging.info("[Memory] 🧠 ChromaDB initialisé en mode Éphémère (RAM).")

    # ==========================================
    # MÉTHODES SYNCHRONES (Bloquantes)
    # ==========================================
    def _add_sync(self, text_content: str, metadata: dict):
        """Ajoute un souvenir mathématiquement dans la RAM."""
        self.memory_counter += 1
        memory_id = f"mem_{self.memory_counter}"
        
        self.collection.add(
            documents=[text_content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        # On ne met pas de print ici pour ne pas spammer la console en plein show

    def _search_sync(self, query_text: str, top_k: int = 2) -> str:
        """Cherche les souvenirs les plus proches sémantiquement."""
        # Sécurité : Si la mémoire est vide, on ne cherche pas
        if self.collection.count() == 0:
            return ""

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        
        # Extraction des documents trouvés
        documents = results.get('documents', [[]])[0]
        if not documents:
            return ""
            
        # On fusionne les souvenirs trouvés en un seul bloc de texte
        return "\n".join([f"- {doc}" for doc in documents])

    # ==========================================
    # WRAPPERS ASYNCHRONES (Pour le Core Engine)
    # ==========================================
    async def add_memory(self, text_content: str, role: str = "user"):
        """Enveloppe asynchrone pour ne pas geler le Cerveau."""
        metadata = {"role": role}
        await asyncio.to_thread(self._add_sync, text_content, metadata)

    async def retrieve_context(self, user_text: str, top_k: int = 2) -> str:
        """Récupère le contexte sans bloquer la boucle OSC/MQTT."""
        return await asyncio.to_thread(self._search_sync, user_text, top_k)