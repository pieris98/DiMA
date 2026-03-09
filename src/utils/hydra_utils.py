import os
from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

def setup_config(config_path, overrides=None):
    # Reset Hydra to avoid conflicts if already initialized
    GlobalHydra.instance().clear()
    # Convert to absolute path so initialize_config_dir works regardless of cwd
    abs_config_path = os.path.abspath(config_path)
    initialize_config_dir(config_dir=abs_config_path, version_base=None)
    # Load the configuration, applying any dot-notation overrides
    cfg = compose(config_name="config", overrides=overrides or [])
    return cfg