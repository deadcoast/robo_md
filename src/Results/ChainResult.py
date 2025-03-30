"""
A class for representing the result of a task chain.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class ChainResult:
    """
    A class for representing the result of a task chain.
    """
    success: bool = False
    metrics: Any = None
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"ChainResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __reduce__(self) -> tuple:
        return (ChainResult, (self.success, self.metrics, self.error, self.timestamp))
    
    def __getstate__(self) -> Dict[str, Any]:
        return {"success": self.success, "metrics": self.metrics, "error": self.error, "timestamp": self.timestamp}
    
    def __setstate__(self, state: Dict[str, Any]):
        self.success = state["success"]
        self.metrics = state["metrics"]
        self.error = state["error"]
        self.timestamp = state["timestamp"]
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ChainResult):
            return False
        return (
            self.success == other.success and
            self.metrics == other.metrics and
            self.error == other.error and
            self.timestamp == other.timestamp
        )
        
    def __hash__(self) -> int:
        return hash((self.success, self.metrics, self.error, self.timestamp))
    
    def __copy__(self) -> 'ChainResult':
        return ChainResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics),
            error=self.error,
            timestamp=self.timestamp
        )
    
    def __deepcopy__(self, memo: Dict[int, Any]) -> 'ChainResult':
        return ChainResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics, memo),
            error=self.error,
            timestamp=self.timestamp
        )
        
    def __copy__(self) -> 'ChainResult':
        return self.__deepcopy__({})
            