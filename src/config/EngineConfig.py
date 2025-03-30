import logging
from dataclasses import dataclass
from typing import Any, Dict

from src.config.EngineConfig import SystemConfig

@dataclass
class EngineConfig:
    config: SystemConfig
    logger: logging.Logger

    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._load_config()

    def _load_config(self):
        self.logger.info("Loading engine configuration.")
        self.logger.debug(f"Engine configuration: {self.config}")
        self.logger.info("Engine configuration loaded.")

    def get_config(self) -> SystemConfig:
        return self.config

    def update_config(self, config: SystemConfig):
        self.config = config
        self.logger.info("Engine configuration updated.")
        self.logger.debug(f"Engine configuration: {self.config}")
        self._load_config()

    def __str__(self) -> str:
        return str(self.config)

    def __repr__(self) -> str:
        return str(self.config)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EngineConfig):
            return False
        return self.config == other.config

    def __ne__(self, other: Any) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash(self.config)

    def __copy__(self) -> "EngineConfig":
        return EngineConfig(self.config)

    def __deepcopy__(self, memo: Dict) -> "EngineConfig":
        return EngineConfig(self.config)

    def __reduce__(self) -> tuple:
        return (EngineConfig, (self.config,))

    def __reduce_ex__(self, protocol: int) -> tuple:
        return (EngineConfig, (self.config,))

    def __getstate__(self) -> Dict[str, Any]:
        return {"config": self.config}

    def __setstate__(self, state: Dict[str, Any]):
        self.config = state["config"]
        self.logger = logging.getLogger(__name__)
        self._load_config()
