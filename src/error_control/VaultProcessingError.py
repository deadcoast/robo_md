"""
A custom exception class for vault processing errors.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class VaultProcessingError(Exception):
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        return f"{self.message} - {self.details}"

    def __repr__(self) -> str:
        return f"{self.message} - {self.details}"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, VaultProcessingError):
            return self.message == other.message and self.details == other.details
        return False

    def __hash__(self) -> int:
        return hash((self.message, self.details))
