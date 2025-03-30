"""
A class for representing the result of a chain execution.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.Results.ExecutionResult import ExecutionResult


@dataclass
class ChainResult:
    """
    A class for representing the result of a chain execution.
    """
    success: bool = False
    metrics: Any = None
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        return f"ChainResult(success={self.success}, metrics={self.metrics}, error={self.error}, timestamp={self.timestamp})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __bool__(self) -> bool:
        return self.success
