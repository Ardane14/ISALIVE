import yaml
import logging
import sys

class ConfigLoader:
    """
    Classe de chargement de configuration à partir d'un fichier YAML.
    filepath : Chemin vers le fichier de configuration YAML (par défaut "conf.yaml").
    """
    
    
    def __init__(self, filepath="conf.yaml"):
        self.filepath = filepath
        self.data = self._load_yaml()

    def _load_yaml(self) -> dict:
        """Lit le fichier YAML de manière sécurisée."""
        try:
            with open(self.filepath, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logging.info(f"Configuration '{self.filepath}' chargée avec succès.")
                return config
        except FileNotFoundError:
            logging.critical(f"ERREUR FATALE: Le fichier {self.filepath} est introuvable.")
            sys.exit(1) # Arrêt d'urgence avant le show
        except yaml.YAMLError as exc:
            logging.critical(f"ERREUR FATALE: Syntaxe YAML invalide dans {self.filepath}\nDétails: {exc}")
            sys.exit(1) # Arrêt d'urgence avant le show

    def get(self, section: str, key: str = None):
        """Méthode utilitaire pour récupérer une valeur (Optionnelle)."""
        try:
            if key:
                return self.data[section][key]
            return self.data[section]
        except KeyError as e:
            logging.error(f"Clé de configuration manquante : {e}")
            return None