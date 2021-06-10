"""This file defines the interface in which all saveable objects
    must follow, note that the PyTorch models, optimizers have this
    implemented already
"""
from typing import Dict
from abc import ABC, abstractmethod


class SaveableInterface(ABC):

    @abstractmethod
    def state_dict(self) -> Dict:
        """Must implement this method to return a dictionary
            for the manager to save, everything in this dictionary
            must be pickleable
        """

    @abstractmethod
    def load_state_dict(self, state_dict: Dict):
        """Given the dictionary format saved, load back up the state
        """


def validate_dict(state_dict, *keys_to_validate):
    """Validates the state_dict to ensure that the state_dict contains
        all the keys we want to use
    """
    for key in keys_to_validate:
        if key not in state_dict:
            raise ValueError(f"Key {key} is not found in state_dict with "
                             f"keys {state_dict.keys()}")
