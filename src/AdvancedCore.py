from src.FeatureCore import FeatureProcessor
from src.AnalyticsCore import EnhancedFeatureSet
from src.AnalyticsCore import TopicModelingEngine
from src.AnalyticsCore import GraphFeatureProcessor
from typing import List, Dict, Any
from src.SystemConfig import SystemConfig
from src.EngineConfig import EngineConfig
from src.AnalyticsCore import EnhancedFeatureSet
from src.AnalyticsCore import TopicModelingEngine
from src.AnalyticsCore import GraphFeatureProcessor
from src.AnalyticsCore import EnhancedFeatureSet


class AdvancedFeatureProcessor(FeatureProcessor):
    """
    A class for processing advanced features.

    Args:
        self: The instance of the AdvancedFeatureProcessor.
        config (SystemConfig): The system configuration.

    Attributes:
        topic_engine (TopicModelingEngine): The topic modeling engine.
        graph_processor (GraphFeatureProcessor): The graph feature processor.

    Methods:
        generate_enhanced_features: Generate enhanced features.
    """
    def __init__(self, config: SystemConfig):
        super().__init__(config)
        self.topic_engine = TopicModelingEngine()
        self.graph_processor = GraphFeatureProcessor()

    async def generate_enhanced_features(
        self, docs: List[Dict[str, Any]]
    ) -> EnhancedFeatureSet:
        embeddings = await self._generate_embeddings(docs)
        topic_features = await self.topic_engine.extract_features(docs)
        graph_features = await self.graph_processor.compute_features(docs)
        metadata_features = await self._process_metadata(docs)

        return EnhancedFeatureSet(
            embeddings=embeddings,
            topic_features=topic_features,
            graph_features=graph_features,
            metadata_features=metadata_features,
        )


class TaskRegistryManager:
    """
    A class for managing tasks.

    Args:
        self: The instance of the TaskRegistryManager.
        config (EngineConfig): The engine configuration.

    Attributes:
        active_tasks (List[Task]): A list of active tasks.
        pending_queue (Queue): A queue of pending tasks.
        completion_log (List[Task]): A list of completed tasks.

    Methods:
        register_next_task: Register and initialize next task in sequence.
    """
    def __init__(self, config: EngineConfig):
        self.active_tasks = []
        self.pending_queue = Queue()
        self.completion_log = []

    async def register_next_task(self) -> TaskRegistration:
        """
        Register and initialize next task in sequence.

        Returns:
            TaskRegistration: The registered task.
        """
        task = await self.pending_queue.get()
        self.active_tasks.append(task)
        return self._generate_task_context(task)
    
    def _generate_task_context(self, task: Task) -> TaskRegistration:
        """
        Generate a task context.

        Args:
            self: The instance of the TaskRegistryManager.
            task (Task): The task to generate context for.

        Returns:
            TaskRegistration: The generated task context.
        """
        return TaskRegistration(task_id=task.id, task=task)
    
    def _log_task_completion(self, task: Task):
        """
        Log the completion of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task (Task): The task to log.
        """
        self.completion_log.append(task)
        self.active_tasks.remove(task)
        self.logger.info(f"Task {task.id} completed.")
        
    def _log_task_error(self, task: Task, error: Exception):
        """
        Log the error of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task (Task): The task to log.
            error (Exception): The error to log.
        """
        self.error_log.append(
            {"task": task.id, "error": str(error), "timestamp": self.get_timestamp()}
        )
        self.active_tasks.remove(task)
        self.logger.error(f"Task {task.id} failed with error: {error}")
        
    def get_timestamp(self) -> str:
        """
        Get the current timestamp.

        Returns:
            str: The current timestamp.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    
    def get_active_tasks(self) -> List[Task]:
        """
        Get the list of active tasks.

        Returns:
            List[Task]: The list of active tasks.
        """
        return self.active_tasks
        
    def get_completion_log(self) -> List[Task]:
        """
        Get the list of completed tasks.

        Returns:
            List[Task]: The list of completed tasks.
        """
        return self.completion_log
        
    def get_error_log(self) -> List[Dict[str, Any]]:
        """
        Get the list of error logs.

        Returns:
            List[Dict[str, Any]]: The list of error logs.
        """
        return self.error_log
        
    def get_pending_tasks(self) -> List[Task]:
        """
        Get the list of pending tasks.

        Returns:
            List[Task]: The list of pending tasks.
        """
        return list(self.pending_queue.queue)
        
    def get_task_registry(self) -> Dict[str, Task]:
        """
        Get the task registry.

        Returns:
            Dict[str, Task]: The task registry.
        """
        return self.task_registry
        
    def get_task_status(self, task_id: str) -> str:
        """
        Get the status of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            str: The status of the task.
        """
        return self.task_registry.get(task_id, {}).get("status", "Unknown")
        
    def get_task_log(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get the log of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            List[Dict[str, Any]]: The log of the task.
        """
        return self.task_registry.get(task_id, {}).get("log", [])
        
    def get_task_errors(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get the errors of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            List[Dict[str, Any]]: The errors of the task.
        """
        return self.task_registry.get(task_id, {}).get("errors", [])
        
    def get_task_metrics(self, task_id: str) -> Dict[str, Any]:
        """
        Get the metrics of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            Dict[str, Any]: The metrics of the task.
        """
        return self.task_registry.get(task_id, {}).get("metrics", {})
        
    def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """
        Get the progress of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            Dict[str, Any]: The progress of the task.
        """
        return self.task_registry.get(task_id, {}).get("progress", {})
        
    def get_task_status(self, task_id: str) -> str:
        """
        Get the status of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            str: The status of the task.
        """
        return self.task_registry.get(task_id, {}).get("status", "Unknown")
        
    def get_task_type(self, task_id: str) -> str:
        """
        Get the type of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            str: The type of the task.
        """
        return self.task_registry.get(task_id, {}).get("type", "Unknown")
        
    def get_task_weight(self, task_id: str) -> float:
        """
        Get the weight of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            float: The weight of the task.
        """
        return self.task_registry.get(task_id, {}).get("weight", 0.0)
        
    def get_task_weight(self, task_id: str) -> float:
        """
        Get the weight of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            float: The weight of the task.
        """
        return self.task_registry.get(task_id, {}).get("weight", 0.0)
        
    def get_task_weight(self, task_id: str) -> float:
        """
        Get the weight of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            float: The weight of the task.
        """
        return self.task_registry.get(task_id, {}).get("weight", 0.0)
    
