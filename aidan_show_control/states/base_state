
from abc import ABC, abstractmethod

class PhaseState(ABC):
    """Classe abstraite : Tous tes états devront respecter ce modèle."""
    
    @abstractmethod
    async def on_enter(self, core):
        pass

    @abstractmethod
    async def on_exit(self, core):
        pass
        
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Retourne le prompt de LM Studio spécifique à cette phase."""
        pass

    @abstractmethod
    async def handle_flag(self, core, flag: str):
        """
        Conditions déclenchée par les flags
        """
        pass