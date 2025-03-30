"""SystemConfig.py

This module contains the SystemConfig class, which represents a system configuration object.
"""

import logging
from typing import Dict, Any

class SystemConfig:
    """
    Represents a system configuration object.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("SystemConfig")
        self._load_config()
        
    def _load_config(self):
        self.logger.info("Loading system configuration.")
        self.logger.debug(f"System configuration: {self.config}")
        self.logger.info("System configuration loaded.")
    
    def get_config(self) -> Dict[str, Any]:
        return self.config
    
    def update_config(self, config: Dict[str, Any]):
        self.config = config
        self.logger.info("System configuration updated.")
        self.logger.debug(f"System configuration: {self.config}")
        self._load_config()
    
    def __str__(self) -> str:
        return str(self.config)
    
    def __repr__(self) -> str:
        return str(self.config)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SystemConfig):
            return False
        return self.config == other.config
    
    def __ne__(self, other: Any) -> bool:
        return not self == other
    
    def __hash__(self) -> int:
        return hash(self.config)
    
    def __copy__(self) -> "SystemConfig":
        return SystemConfig(self.config)
    
    def __deepcopy__(self, memo: Dict) -> "SystemConfig":
        return SystemConfig(self.config)
    
    def __reduce__(self) -> tuple:
        return (SystemConfig, (self.config,))
    
    def __getstate__(self) -> Dict[str, Any]:
        return {"config": self.config}
    
    def __setstate__(self, state: Dict[str, Any]):
        self.config = state["config"]
        self.logger = logging.getLogger("SystemConfig")
        self._load_config()