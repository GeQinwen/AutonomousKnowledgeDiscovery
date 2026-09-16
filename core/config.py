"""Configuration management for AutoKD."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration management."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._load_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            return self._default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration if file doesn't exist."""
        return {
            "goals": {
                "primary": "general discovery"  # Domain-agnostic default
            },
            "agents": {
                "generator": {"batch_size": 5},
                "evaluator": {"min_p_value": 0.05},
                "critic": {"min_insight_score": 0.6}
            },
            "memory": {
                "storage_path": "./data/memory/insight_graph.json"
            }
        }
    
    def _load_env_overrides(self):
        """Override config with environment variables."""
        if os.getenv("OPENAI_API_KEY"):
            self.config.setdefault("api", {})["openai_key"] = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("PRIMARY_GOAL"):
            self.config.setdefault("goals", {})["primary"] = os.getenv("PRIMARY_GOAL")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot-separated path."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        return self.config.get("agents", {}).get(agent_name, {})
    
    def get_goal(self) -> str:
        """Get the primary discovery goal."""
        return self.config.get("goals", {}).get("primary", "general discovery")
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.config.get("llm", {
            "provider": "ollama",
            "model": "llama3",
            "base_url": "http://localhost:11434"
        })


# Lazily created instance for get_config()
_config_instance = None


def get_config(config_path: str = "config/config.yaml") -> Config:
    """Get global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


# Global config instance
config = Config()

