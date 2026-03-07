"""
Persistence module for saving and loading simulation states.
"""

import json
import logging
from typing import Tuple

from core.data_structures import CharacterState, WorldState

logger = logging.getLogger(__name__)

def save_simulation_state(
    character_state: CharacterState,
    world_state: WorldState,
    filepath: str
) -> bool:
    """
    Save the current simulation state to a JSON file.
    
    Parameters
    ----------
    character_state : CharacterState
    world_state : WorldState
    filepath : str
    
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    data = {
        "character_state": character_state.to_dict(),
        "world_state": world_state.to_dict()
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"State saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save state to {filepath}: {e}")
        return False

def load_simulation_state(filepath: str) -> Tuple[CharacterState, WorldState]:
    """
    Load simulation state from a JSON file.
    
    Parameters
    ----------
    filepath : str
    
    Returns
    -------
    Tuple[CharacterState, WorldState]
        Loaded states. Raises exception if failure.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        char_state = CharacterState.from_dict(data["character_state"])
        world_state = WorldState.from_dict(data["world_state"])
        
        logger.info(f"State loaded from {filepath}")
        return char_state, world_state
    except Exception as e:
        logger.error(f"Failed to load state from {filepath}: {e}")
        raise
