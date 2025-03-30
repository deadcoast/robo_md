
from dataclasses import dataclass
from typing import List, Dict, Any
from src.SystemConfig import SystemConfig
from queue import PriorityQueue
from src.ChainMetrics import ChainMetrics
from src.ChainResult import ChainResult
from src.ChainResult import ExecutionResult


@dataclass
class TaskChainConfig:
    """
    A class for configuring task chains.

    Args:
        self: The instance of the TaskChainConfig.
        chain_id (str): The ID of the task chain.
        priority (int): The priority of the task chain.
        dependencies (List[str]): A list of dependencies for the task chain.
        resource_requirements (Dict[str, float]): A dictionary of resource requirements for the task chain.

    Attributes:
        chain_id (str): The ID of the task chain.
        priority (int): The priority of the task chain.
        dependencies (List[str]): A list of dependencies for the task chain.
        resource_requirements (Dict[str, float]): A dictionary of resource requirements for the task chain.
    """
    chain_id: str
    priority: int
    dependencies: List[str]
    resource_requirements: Dict[str, float]


class ChainManager:
    """
    A class for managing task chains.

    Args:
        self: The instance of the ChainManager.
        config (SystemConfig): The system configuration.

    Attributes:
        active_chains (List[TaskChainConfig]): A list of active task chains.
        execution_queue (PriorityQueue): A priority queue of task chains.
        completion_registry (Dict[str, TaskChainConfig]): A dictionary of completed task chains.

    Methods:
        process_next_chain: Process the next task chain in the queue.
    """
    def __init__(self, config: SystemConfig):
        self.active_chains = []
        self.execution_queue = PriorityQueue()
        self.completion_registry = {}

    async def process_next_chain(self) -> ChainResult:
        chain = await self.execution_queue.get()
        return await self._execute_chain(chain)


class ChainExecutor:
    """
    A class for executing task chains.

    Args:
        self: The instance of the ChainExecutor.

    Attributes:
        validator (ChainValidator): The chain validator.
        resource_manager (ResourceManager): The resource manager.
        monitor (ExecutionMonitor): The execution monitor.

    Methods:
        execute_chain: Execute a task chain.
    """
    def __init__(self):
        self.validator = ChainValidator()
        self.resource_manager = ResourceManager()
        self.monitor = ExecutionMonitor()

    async def execute_chain(self, chain_config: TaskChainConfig) -> ExecutionResult:
        try:
            # Initialize chain context
            context = await self._prepare_chain_context(chain_config)

            # Execute with monitoring
            with self.monitor.track_chain():
                result = await self._process_chain(context)

            return ChainResult(
                success=True, metrics=self._compute_chain_metrics(result)
            )

        except Exception as e:
            return ChainResult(success=False, error=str(e))


@dataclass
class ChainMetrics:
    """
    A class for tracking metrics of a task chain.

    Args:
        self: The instance of the ChainMetrics.
        chain_id (str): The ID of the task chain.
        execution_time (float): The execution time of the task chain.
        task_completion (Dict[str, float]): A dictionary of task completion metrics.
        resource_allocation (Dict[str, float]): A dictionary of resource allocation metrics.
        error_registry (List[str]): A list of error registry.

    Attributes:
        chain_id (str): The ID of the task chain.
        execution_time (float): The execution time of the task chain.
        task_completion (Dict[str, float]): A dictionary of task completion metrics.
        resource_allocation (Dict[str, float]): A dictionary of resource allocation metrics.
        error_registry (List[str]): A list of error registry.
    """
    chain_id: str
    execution_time: float
    task_completion: Dict[str, float]
    resource_allocation: Dict[str, float]
    error_registry: List[str] = field(default_factory=list)
