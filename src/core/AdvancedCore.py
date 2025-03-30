import contextlib
import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Any, Dict, List, Optional, TypedDict

from src.AnalyticsCore import (
    EnhancedFeatureSet,
    GraphFeatureProcessor,
    TopicModelingEngine,
)
from src.engines.EngineConfig import EngineConfig
from src.FeatureCore import FeatureProcessor
from src.SystemConfig import SystemConfig


# Define TypedDicts for metadata structure
class TimestampData(TypedDict):
    earliest: Optional[datetime]
    latest: Optional[datetime]
    distribution: Dict[str, int]


class SizeStatistics(TypedDict):
    min_size: float
    max_size: int
    avg_size: float
    total_size: int


class MetadataFeatures(TypedDict):
    document_count: int
    categories: Dict[str, int]
    authors: Dict[str, int]
    timestamps: TimestampData
    size_statistics: SizeStatistics
    tag_frequency: Dict[str, int]


@dataclass
class Task:
    id: str
    type: str
    status: str = "pending"
    weight: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    registration: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@dataclass
class TaskRegistration:
    task_id: str
    task: Task


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
        self.logger = logging.getLogger(__name__)

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

    async def _generate_embeddings(
        self, docs: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """
        Generate embeddings for the input documents.

        Args:
            self: The instance of the AdvancedFeatureProcessor.
            docs: The documents to generate embeddings for.

        Returns:
            List[List[float]]: The generated embeddings.
        """
        embeddings = []
        embedding_size = 768  # Standard embedding dimension

        # Initialize logger if not already present
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(__name__)

        for doc in docs:
            # Extract text content from document
            text = doc.get("content", "")
            if not text and "title" in doc:
                text = doc["title"]

            # Normalize text
            text = text.lower()

            # Generate a simple embedding based on character frequencies
            # This is a simplified approach; in a real-world scenario, you'd use
            # a proper embedding model like sentence-transformers or similar
            char_freqs: Dict[str, int] = {}
            for char in text:
                if char in char_freqs:
                    char_freqs[char] += 1
                else:
                    char_freqs[char] = 1

            # Create a simple embedding vector
            # In a real implementation, this would use a proper embedding model

            # Generate a deterministic but somewhat distributed embedding
            embedding = [0.0] * embedding_size
            if text:
                for i, char in enumerate(text):
                    # Create a reproducible hash for each character position (not for security purposes)
                    hash_val = int(
                        hashlib.md5(
                            f"{char}_{i}".encode(), usedforsecurity=False
                        ).hexdigest(),
                        16,
                    )
                    # Distribute the value across the embedding
                    positions = [
                        hash_val % embedding_size,
                        (hash_val // 256) % embedding_size,
                        (hash_val // 65536) % embedding_size,
                    ]
                    for pos in positions:
                        embedding[pos] += char_freqs.get(char, 0) * 0.01

                # Normalize the embedding
                magnitude = math.sqrt(sum(x**2 for x in embedding))
                if magnitude > 0:
                    embedding = [x / magnitude for x in embedding]

            embeddings.append(embedding)

        self.logger.info(f"Generated {len(embeddings)} document embeddings")
        return embeddings

    async def _process_metadata(self, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process metadata from the input documents.

        Args:
            self: The instance of the AdvancedFeatureProcessor.
            docs: The documents to process metadata from.

        Returns:
            Dict[str, Any]: The processed metadata.
        """
        metadata_features: MetadataFeatures = {
            "document_count": len(docs),
            "categories": {},
            "authors": {},
            "timestamps": {"earliest": None, "latest": None, "distribution": {}},
            "size_statistics": {
                "min_size": float("inf"),
                "max_size": 0,
                "avg_size": 0,
                "total_size": 0,
            },
            "tag_frequency": {},
        }

        # Initialize logger if not already present
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(__name__)

        # Process each document's metadata
        for doc in docs:
            # Extract and process categories
            if "category" in doc:
                category = doc["category"]
                if category in metadata_features["categories"]:
                    metadata_features["categories"][category] += 1
                else:
                    metadata_features["categories"][category] = 1

            # Extract and process authors
            if "author" in doc:
                author = doc["author"]
                if author in metadata_features["authors"]:
                    metadata_features["authors"][author] += 1
                else:
                    metadata_features["authors"][author] = 1

            # Process timestamps
            if "timestamp" in doc:
                timestamp = doc["timestamp"]
                # Format might vary, assuming ISO format for simplicity
                with contextlib.suppress(ValueError, AttributeError):
                    # Convert to datetime object if string
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )

                    # Update earliest/latest timestamps
                    if (
                        metadata_features["timestamps"]["earliest"] is None
                        or timestamp < metadata_features["timestamps"]["earliest"]
                    ):
                        metadata_features["timestamps"]["earliest"] = timestamp

                    if (
                        metadata_features["timestamps"]["latest"] is None
                        or timestamp > metadata_features["timestamps"]["latest"]
                    ):
                        metadata_features["timestamps"]["latest"] = timestamp

                    # Add to distribution (by month for visualization)
                    month_key = f"{timestamp.year}-{timestamp.month:02d}"
                    if month_key in metadata_features["timestamps"]["distribution"]:
                        metadata_features["timestamps"]["distribution"][month_key] += 1
                    else:
                        metadata_features["timestamps"]["distribution"][month_key] = 1
            # Process document size
            content = doc.get("content", "")
            doc_size = len(content)
            metadata_features["size_statistics"]["total_size"] += doc_size
            metadata_features["size_statistics"]["min_size"] = min(
                metadata_features["size_statistics"]["min_size"], doc_size
            )
            metadata_features["size_statistics"]["max_size"] = max(
                metadata_features["size_statistics"]["max_size"], doc_size
            )

            # Process tags
            if "tags" in doc and isinstance(doc["tags"], list):
                for tag in doc["tags"]:
                    if tag in metadata_features["tag_frequency"]:
                        metadata_features["tag_frequency"][tag] += 1
                    else:
                        metadata_features["tag_frequency"][tag] = 1

        # Calculate average size
        if docs:
            metadata_features["size_statistics"]["avg_size"] = metadata_features[
                "size_statistics"
            ]["total_size"] / len(docs)

        # Fix if no documents had content
        if metadata_features["size_statistics"]["min_size"] == float("inf"):
            metadata_features["size_statistics"]["min_size"] = 0

        self.logger.info(f"Processed metadata for {len(docs)} documents")
        return {"metadata_features": metadata_features}


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
        self.active_tasks: List[Task] = []
        self.pending_queue: Queue = Queue()
        self.completion_log: List[Task] = []
        self.task_registry: Dict[str, Dict[str, Any]] = {}
        self.error_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

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
        self.logger.info(f"Generating task context for task {task.id}.")
        self.logger.debug(f"Task details: {task}")
        self.logger.debug(f"Task context: {task.context}")
        self.logger.debug(f"Task registration: {task.registration}")
        return TaskRegistration(task_id=task.id, task=task)

    def _log_task_completion(self, task: Task) -> None:
        """
        Log the completion of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task (Task): The task to log.
        """
        self.completion_log.append(task)
        self.active_tasks.remove(task)
        self.logger.info(f"Task {task.id} completed.")

    def _log_task_error(self, task: Task, error: Exception) -> None:
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
        self.logger.info("Getting current timestamp.")
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_active_tasks(self) -> List[Task]:
        """
        Get the list of active tasks.

        Returns:
            List[Task]: The list of active tasks.
        """
        self.active_tasks = sorted(
            self.active_tasks, key=lambda task: task.timestamp, reverse=False
        )
        return self.active_tasks

    def get_completion_log(self) -> List[Task]:
        """
        Get the list of completed tasks.

        Returns:
            List[Task]: The list of completed tasks.
        """
        self.completion_log = sorted(
            self.completion_log, key=lambda task: task.timestamp, reverse=False
        )
        return self.completion_log

    def get_error_log(self) -> List[Dict[str, Any]]:
        """
        Get the list of error logs.

        Returns:
            List[Dict[str, Any]]: The list of error logs.
        """
        self.error_log.sort(key=lambda log: log["timestamp"], reverse=True)
        self.error_log.reverse()
        return self.error_log

    def get_task_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the task registry.

        Returns:
            Dict[str, Dict[str, Any]]: The task registry.
        """
        self.task_registry.sort(key=lambda task: task.timestamp, reverse=True)
        self.task_registry.reverse()
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
        # Check active tasks first for more up-to-date status
        for task in self.active_tasks:
            if task.id == task_id:
                return task.status

        # Then check the task registry
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
        return self._extracted_from_get_task_errors_12(task_id, "log")

    def get_task_errors(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get the errors of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            List[Dict[str, Any]]: The errors of the task.
        """
        return self._extracted_from_get_task_errors_12(task_id, "errors")

    # TODO Rename this here and in `get_task_log` and `get_task_errors`
    def _extracted_from_get_task_errors_12(self, task_id, arg1):
        self.task_registry.get(task_id, {}).get(arg1, []).sort(
            key=lambda log: log["timestamp"], reverse=True
        )
        self.task_registry.get(task_id, {}).get(arg1, []).reverse()
        return self.task_registry.get(task_id, {}).get(arg1, [])

    def get_task_metrics(self, task_id: str) -> Dict[str, Any]:
        """
        Get the metrics of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            Dict[str, Any]: The metrics of the task.
        """
        return self._update_tasks(task_id, "metrics")

    def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """
        Get the progress of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            Dict[str, Any]: The progress of the task.
        """
        return self._update_tasks(task_id, "progress")

    # TODO Rename this here and in `get_task_metrics` and `get_task_progress`
    def _update_tasks(self, task_id, arg1):
        """
        Get the progress of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.
            arg1 (str): The key to access in the task registry.

        Returns:
            Dict[str, Any]: The progress of the task.
        """
        self.task_registry.get(task_id, {}).get(arg1, {}).sort(
            key=lambda metric: metric["timestamp"], reverse=True
        )
        self.task_registry.get(task_id, {}).get(arg1, {}).reverse()
        return self.task_registry.get(task_id, {}).get(arg1, {})

    def get_task_type(self, task_id: str) -> str:
        """
        Get the type of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            str: The type of the task.
        """
        return self._extracted_from_get_task_weight_12(
            "Getting task type for task ", task_id, "type", "Unknown"
        )

    def get_task_weight(self, task_id: str) -> float:
        """
        Get the weight of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.

        Returns:
            float: The weight of the task.
        """
        return self._extracted_from_get_task_weight_12(
            "Getting task weight for task ", task_id, "weight", 0.0
        )

    # TODO Rename this here and in `get_task_type` and `get_task_weight`
    def _extracted_from_get_task_weight_12(self, arg0, task_id, arg2, arg3):
        self.logger.info(f"{arg0}{task_id}.")
        self.task_registry.get(task_id, {}).get(arg2, arg3).sort(
            key=lambda type: type["timestamp"], reverse=True
        )
        self.task_registry.get(task_id, {}).get(arg2, arg3).reverse()
        return self.task_registry.get(task_id, {}).get(arg2, arg3)

    def add_task(self, task: Task) -> None:
        """
        Add a task to the pending queue.

        Args:
            self: The instance of the TaskRegistryManager.
            task (Task): The task to add.
        """
        self.logger.info(f"Adding task {task.id} to pending queue.")
        try:
            self.pending_queue.put_nowait(task)
            # Register the task in the task registry
            self.task_registry[task.id] = {
                "task": task,
                "status": task.status,
                "type": task.type,
                "weight": task.weight,
                "log": [],
                "errors": [],
                "metrics": {},
                "progress": {
                    "percent_complete": 0,
                    "current_step": "pending",
                    "total_steps": 0,
                    "created_at": self.get_timestamp(),
                    "updated_at": self.get_timestamp(),
                },
            }
            self.logger.info(f"Added task {task.id} to pending queue")
        except Exception as e:
            self.logger.error(f"Failed to add task {task.id}: {str(e)}")
            # Add to error log even though task isn't in active tasks
            self.error_log.append(
                {"task": task.id, "error": str(e), "timestamp": self.get_timestamp()}
            )

    def update_task_progress(
        self,
        task_id: str,
        percent_complete: float,
        current_step: str,
        total_steps: int = 0,
    ) -> None:
        """
        Update the progress of a task.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.
            percent_complete (float): The percentage complete (0-100).
            current_step (str): The current step description.
            total_steps (int, optional): The total number of steps. Defaults to 0.
        """
        self.logger.info(
            f"Updating task {task_id} progress: {percent_complete}% - {current_step}"
        )
        # Convert list to Queue type-safely
        if task_id in self.task_registry:
            self.task_registry[task_id]["progress"] = {
                "percent_complete": max(
                    0, min(100, percent_complete)
                ),  # Ensure between 0-100
                "current_step": current_step,
                "total_steps": total_steps,
                "updated_at": self.get_timestamp(),
            }
            self.logger.info(
                f"Updated task {task_id} progress: {percent_complete}% - {current_step}"
            )

    # TODO Rename this here and in `update_task_progress`
    def _extracted_from_update_task_progress_12(
        self, task_id, percent_complete, current_step, total_steps
    ):
        self.logger.info(
            f"Updating task {task_id} progress: {percent_complete}% - {current_step}"
        )
        # Convert list to Queue type-safely
        if task_id in self.task_registry:
            self.task_registry[task_id]["progress"] = {
                "percent_complete": max(
                    0, min(100, percent_complete)
                ),  # Ensure between 0-100
                "current_step": current_step,
                "total_steps": total_steps,
                "updated_at": self.get_timestamp(),
            }
            self.logger.info(
                f"Updated task {task_id} progress: {percent_complete}% - {current_step}"
            )

    def pending_tasks(self) -> List[str]:
        """
        Get a list of pending tasks.

        Returns:
            List[str]: A list of pending task IDs.
        """
        self.logger.info("Getting pending tasks.")
        return [task.id for task in self.active_tasks if task.status == "pending"]

    def complete_task(self, task_id: str, result: Any = None) -> None:
        """
        Mark a task as completed.

        Args:
            self: The instance of the TaskRegistryManager.
            task_id (str): The ID of the task.
            result (Any, optional): The result of the task. Defaults to None.
        """
        self.logger.info(f"Marking task {task_id} as completed.")
        # Find the task in active tasks
        for task in self.active_tasks:
            if task.id == task_id:
                task.status = "completed"
                self._log_task_completion(task)
                # Update registry
                if task_id in self.task_registry:
                    self._extracted_from_complete_task_18(task_id, result)
                self.logger.info(f"Marked task {task_id} as completed.")
                return

        # If we get here, the task was not found in the active tasks
        self.logger.warning(f"Task {task_id} not found in active tasks.")

        # Update the registry anyway if it exists
        if task_id in self.task_registry:
            self._extracted_from_complete_task_18(task_id, result)
        self._temporary_que_list(task_id)
        # Remove the task from error log
        self.error_log = [log for log in self.error_log if log["task"] != task_id]
        self.logger.info(f"Removed task {task_id} from error log.")

        # Remove the task from completion log
        self.completion_log = [log for log in self.completion_log if log.id != task_id]
        self.logger.info(f"Removed task {task_id} from completion log.")

        # Remove the task from active tasks
        self.active_tasks = [task for task in self.active_tasks if task.id != task_id]
        self.logger.info(f"Removed task {task_id} from active tasks.")

        self._temporary_que_list(task_id)

    def _temporary_que_list(self, task_id):
        # Create a temporary list to store all tasks except the one to remove
        result = []
        # Empty the queue
        while not self.pending_queue.empty():
            t = self.pending_queue.get()
            if t.id != task_id:
                result.append(t)

        # Re-add tasks to the queue except the removed one
        for t in result:
            self.pending_queue.put(t)
        self.logger.info(f"Removed task {task_id} from pending queue.")

        # Remove the task from task registry
        # Convert list to Queue type-safely
        if task_id in self.task_registry:
            del self.task_registry[task_id]
            self.logger.info(f"Removed task {task_id} from task registry.")

        return result

    # TODO Rename this here and in `complete_task`
    def _extracted_from_complete_task_18(self, task_id, result):
        self.task_registry[task_id]["status"] = "completed"
        self.task_registry[task_id]["progress"]["percent_complete"] = 100
        self.task_registry[task_id]["progress"]["current_step"] = "completed"
        self.task_registry[task_id]["progress"]["updated_at"] = self.get_timestamp()
        if result is not None:
            self.task_registry[task_id]["result"] = result
