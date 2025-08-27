from box import Box
import json
from pathlib import Path

def load_config(path: Path = Path("config.json")) -> Box:
    """
    Load project configuration from a JSON file.

    This function loads the configuration from the given JSON file
    and wraps it in a `Box` object, allowing both key-based and 
    dot-notation access (e.g., `config["key"]` or `config.key`).

    Args:
        path (Path, optional): Path to the configuration JSON file. 
                               Defaults to "config.json" in the project root.

    Returns:
        Box: A dot-accessible configuration object containing all 
             settings defined in the JSON file.

    Usage:
        >>> CONFIG = load_config()
        >>> print(CONFIG.data_upload.root_dir)
        >>> print(CONFIG.model_training.TRAIN_FILE_NAME)

    Notes:
        - Keep environment-specific variables (like secrets, DB credentials) 
          outside this file, preferably in environment variables.
        - This function is safe for production use since it:
            * Ensures structured config
            * Supports dot-access
            * Avoids hardcoding paths
    """
    
    with open(path, "r") as f:
        return Box(json.load(f))  # dot-accessible JSON config

def load_constants(path: Path = Path("parms.json")) -> Box:
    with open(path, "r") as f:
        return Box(json.load(f))  # dot-accessible JSON config
    
PARMS = load_constants()

# Global config object
CONFIG = load_config()
