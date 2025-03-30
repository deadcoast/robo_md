"""
A class for representing the result of an execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime

@dataclass
class ExecutionResult:
    """
    A class for representing the result of an execution.
    """
    success: bool = False
    metrics: Any = None
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"ExecutionResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __reduce__(self) -> tuple:
        return (ExecutionResult, (self.success, self.metrics, self.error, self.timestamp))
    
    def __getstate__(self) -> Dict[str, Any]:
        return {"success": self.success, "metrics": self.metrics, "error": self.error, "timestamp": self.timestamp}
    
    def __setstate__(self, state: Dict[str, Any]):
        self.success = state["success"]
        self.metrics = state["metrics"]
        self.error = state["error"]
        self.timestamp = state["timestamp"]
    
    def __copy__(self) -> 'ExecutionResult':
        return ExecutionResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics),
            error=self.error,
            timestamp=self.timestamp
        )
    
    def __deepcopy__(self, memo: Dict[int, Any]) -> 'ExecutionResult':
        return ExecutionResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics, memo),
            error=self.error,
            timestamp=self.timestamp
        )
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ExecutionResult):
            return False
        return (
            self.success == other.success and
            self.metrics == other.metrics and
            self.error == other.error and
            self.timestamp == other.timestamp
        )
    
    def __ne__(self, other: Any) -> bool:
        return not self == other
    
    def __hash__(self) -> int:
        return hash((self.success, self.metrics, self.error, self.timestamp))
    
    def __bool__(self) -> bool:
        return self.success
    
    def __len__(self) -> int:
        return len(self.metrics)
    
    def __iter__(self) -> Iterator[Any]:
        return iter(self.metrics)
    
    def __contains__(self, item: Any) -> bool:
        return item in self.metrics
    
    def __getitem__(self, key: Any) -> Any:
        return self.metrics[key]
    
    def __setitem__(self, key: Any, value: Any) -> None:
        self.metrics[key] = value
    
    def __delitem__(self, key: Any) -> None:
        del self.metrics[key]
    
    def __repr__(self) -> str:
        return f"ExecutionResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"
    
    def __str__(self) -> str:
        return f"ExecutionResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"
    
    def __reduce__(self) -> tuple:
        return (ExecutionResult, (self.success, self.metrics, self.error, self.timestamp))
    
    def __getstate__(self) -> Dict[str, Any]:
        return {"success": self.success, "metrics": self.metrics, "error": self.error, "timestamp": self.timestamp}
    
    def __setstate__(self, state: Dict[str, Any]):
        self.success = state["success"]
        self.metrics = state["metrics"]
        self.error = state["error"]
        self.timestamp = state["timestamp"]
    
    def __copy__(self) -> 'ExecutionResult':
        return ExecutionResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics),
            error=self.error,
            timestamp=self.timestamp
        )
    
    def __deepcopy__(self, memo: Dict[int, Any]) -> 'ExecutionResult':
        return ExecutionResult(
            success=self.success,
            metrics=copy.deepcopy(self.metrics, memo),
            error=self.error,
            timestamp=self.timestamp
        )