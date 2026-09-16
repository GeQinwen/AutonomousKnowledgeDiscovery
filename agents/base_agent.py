"""Base class for all agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict
from core.config import config


class BaseAgent(ABC):
    """Base class for all AutoKD agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.config = config.get_agent_config(name)
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process input and return output. Must be implemented by subclasses."""
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get agent-specific configuration value."""
        return self.config.get(key, default)





