"""
State Manager.

This module provides a comprehensive state manager for managing the state of the application.
"""

import logging
import pickle
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, Optional, Type, Union


class StateManager:
    """
    A comprehensive state manager for managing the state of the application.
    """

    def __init__(self):
        """
        Initialize the state manager with empty state dictionaries and set up logging.
        """
        self.states: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.info("StateManager initialized.")

    def __enter__(self) -> "StateManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.logger.info("StateManager exiting.")

    def add_state(self, state_name: str, state: Any) -> None:
        """
        Add a state to the state manager.

        Args:
            state_name (str): The name of the state.
            state (Any): The state to add.
        """
        self.states[state_name] = state
        self.logger.info(f"State '{state_name}' added.")

    def get_state(self, state_name: str) -> Optional[Any]:
        """
        Get a state from the state manager.

        Args:
            state_name (str): The name of the state.

        Returns:
            Optional[Any]: The state if found, otherwise None.
        """
        return self.states.get(state_name)

    def remove_state(self, state_name: str) -> None:
        """
        Remove a state from the state manager.

        Args:
            state_name (str): The name of the state to remove.
        """
        if state_name in self.states:
            del self.states[state_name]
            self.logger.info(f"State '{state_name}' removed.")
        else:
            self.logger.warning(f"State '{state_name}' not found.")

    def save_state(self, state_name: str, filepath: Union[str, Path]) -> None:
        """
        Save a state to a file.

        Args:
            state_name (str): The name of the state to save.
            filepath (Union[str, Path]): The path to save the state to.
        """
        state = self.get_state(state_name)
        if state is None:
            self.logger.warning(f"State '{state_name}' not found.")
            return

        # Save state to file
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        self.logger.info(f"State '{state_name}' saved to '{filepath}'.")

    def load_state(self, state_name: str, filepath: Union[str, Path]) -> None:
        """
        Load a state from a file.

        Args:
            state_name (str): The name of the state to load.
            filepath (Union[str, Path]): The path to load the state from.
        """
        # Load state from file
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.add_state(state_name, state)
        self.logger.info(f"State '{state_name}' loaded from '{filepath}'.")

    def __reduce__(self) -> "StateManager":
        return (StateManager, (self.states))

    def __getstate__(self) -> Dict[str, Any]:
        return self.states

    def __setstate__(self, states: Dict[str, Any]) -> None:
        self.states = states
