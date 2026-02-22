"""
Model Resolver - Centralized LLM configuration and loading
"""

import os
from pathlib import Path
from enum import Enum
from typing import Optional, Dict
# Provider-agnostic import
from langchain_google_genai import ChatGoogleGenerativeAI


class ModelRole(str, Enum):
    """Predefined model roles"""
    SMART = "smart"
    WORKER = "worker"
    BETTER = "better"
    DUMB = "dumb"
    ARCHITECT = "architect"
    SPECIALIST = "specialist"
    TECH_LEAD = "tech_lead"
    INTEGRATOR = "integrator"
    JUNIOR_DEV = "junior_dev"
    DECOMPOSER = "decomposer"
    CONTRACT_GENERATOR = "contract_generator"


class ModelResolver:
    """Singleton resolver for model configuration"""
    
    _instance = None
    _config_loaded = False
    _model_map: Dict[str, str] = {}
    _temperature_map: Dict[str, float] = {}
    _model_cache: Dict[tuple, ChatGoogleGenerativeAI] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config_loaded:
            self._load_config()
            self._config_loaded = True
    
    def _find_config_file(self) -> Optional[Path]:
        """Search for model_config.txt"""
        search_paths = [
            Path.cwd() / "model_config.txt",
            Path(__file__).parent / "model_config.txt",
            Path(__file__).parent.parent / "model_config.txt",
        ]
        
        if env_path := os.getenv("MODEL_CONFIG_PATH"):
            search_paths.insert(0, Path(env_path))
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_config(self):
        """Load model configuration from model_config.txt"""
        config_path = self._find_config_file()
        
        if config_path is None:
            self._use_defaults()
            return
        
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if key.startswith('temperature_'):
                        role = key.replace('temperature_', '')
                        try:
                            self._temperature_map[role] = float(value)
                        except ValueError:
                            pass
                    else:
                        self._model_map[key] = value
            
        except Exception as e:
            self._use_defaults()
    
    def _use_defaults(self):
        """Fallback defaults"""
        self._model_map = {
            "smart": "LLM-2.5-pro",
            "better": "LLM-2.5-pro",
            "worker": "LLM-2.5-flash",
            "dumb": "LLM-2.5-flash",
            "architect": "LLM-2.5-pro",
            "specialist": "LLM-2.5-pro",
            "tech_lead": "LLM-2.5-pro",
            "integrator": "LLM-2.5-flash",
            "junior_dev": "LLM-2.5-flash",
            "decomposer": "LLM-2.5-pro",
            "contract_generator": "LLM-2.5-flash",
        }
        self._temperature_map = {"smart": 0.0, "worker": 0.0, "decomposer": 0.1}
    
    def resolve_model_name(self, role: str) -> str:
        """Resolve role to model name"""
        role = role.lower().replace("-", "_")
        if role in self._model_map:
            return self._model_map[role]
        # Fallback to a system default if not mapped
        return self._model_map.get("smart", os.getenv("DEFAULT_MODEL_NAME", "default-model"))
    
    def get_default_temperature(self, role: str) -> float:
        """Get default temperature for role"""
        role = role.lower().replace("-", "_")
        return self._temperature_map.get(role, 0.0)
    
    def create_model(
        self,
        role: str,
        temperature: Optional[float] = None,
        **kwargs
    ) -> ChatGoogleGenerativeAI:
        """Create model instance for role"""
        if isinstance(role, ModelRole):
            role = role.value
        
        model_name = self.resolve_model_name(role)
        
        if temperature is None:
            temperature = self.get_default_temperature(role)
        
        cache_key = (model_name, temperature, tuple(sorted(kwargs.items())))
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]
        
        full_model_name = f"models/{model_name}"
        model = ChatGoogleGenerativeAI(
            model=full_model_name,
            temperature=temperature,
            **kwargs
        )
        
        self._model_cache[cache_key] = model
        return model


_resolver = ModelResolver()


def get_model(
    role: str,
    temperature: Optional[float] = None,
    **kwargs
) -> ChatGoogleGenerativeAI:
    """Get model instance for specific role"""
    return _resolver.create_model(role, temperature, **kwargs)
