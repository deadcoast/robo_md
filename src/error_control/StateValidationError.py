"""StateValidationError class."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class StateValidationError(Exception):
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, message: str, details: Dict[str, Any] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.message} ({self.details})"

    def __repr__(self) -> str:
        return f"StateValidationError(message={self.message}, details={self.details})"

    def __reduce__(self) -> "StateValidationError":
        return (StateValidationError, (self.message, self.details))

    def __getstate__(self) -> Dict[str, Any]:
        return {"message": self.message, "details": self.details}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.message = state["message"]
        self.details = state["details"]
        super().__init__(self.message)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateValidationError):
            return False
        return self.message == other.message and self.details == other.details

    def __hash__(self) -> int:
        return hash((self.message, self.details))
