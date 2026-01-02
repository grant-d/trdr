"""SICA path resolution utilities.

Provides functions for resolving paths within the .sica directory structure.
All paths are relative to current working directory.

Directory Structure:
    .sica/                          # Root (created on first use)
    ├── .gitignore                  # Contains: **/runs/
    └── configs/                    # Config folders
        └── <name>/                 # Individual config
            ├── config.json         # User configuration
            ├── state.json          # Run state (single source of truth)
            └── runs/               # Run archives (gitignored)
                └── run_YYYYMMDD_HHMMSS/
                    ├── journal.md
                    └── iteration_N/
"""

from pathlib import Path


def get_sica_root() -> Path:
    """Get .sica directory, create if needed.

    Returns:
        Path to .sica directory in current working directory
    """
    root = Path(".sica")
    root.mkdir(exist_ok=True)
    return root


def get_configs_dir() -> Path:
    """Get .sica/configs directory, create if needed.

    Returns:
        Path to .sica/configs directory
    """
    configs = get_sica_root() / "configs"
    configs.mkdir(exist_ok=True)
    return configs


def get_config_dir(name: str) -> Path:
    """Get config folder by name.

    Does NOT create the directory - use for checking existence.

    Args:
        name: Config name (e.g., 'btc-1h', 'api-tests')

    Returns:
        Path to .sica/configs/<name>/
    """
    return get_configs_dir() / name


def get_config_file(name: str) -> Path:
    """Get config.json path for a config.

    Does NOT create the file - use for checking existence.

    Args:
        name: Config name

    Returns:
        Path to .sica/configs/<name>/config.json
    """
    return get_config_dir(name) / "config.json"


def get_state_file(name: str) -> Path:
    """Get state.json path for a config.

    State file is the single source of truth for run state.
    Check status field to determine if run is active or complete.

    Args:
        name: Config name

    Returns:
        Path to .sica/configs/<name>/state.json
    """
    return get_config_dir(name) / "state.json"


def get_runs_dir(name: str) -> Path:
    """Get runs directory for a config, create if needed.

    Run archives are stored here and gitignored.

    Args:
        name: Config name

    Returns:
        Path to .sica/configs/<name>/runs/
    """
    runs = get_config_dir(name) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return runs


def get_run_dir(name: str, run_id: str) -> Path:
    """Get specific run directory, create if needed.

    Each run gets a timestamped directory containing iteration archives.

    Args:
        name: Config name
        run_id: Run ID in YYYYMMDD_HHMMSS format

    Returns:
        Path to .sica/configs/<name>/runs/run_<run_id>/
    """
    run_dir = get_runs_dir(name) / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def list_configs() -> list[str]:
    """List available config names.

    Only returns configs that have a config.json file.

    Returns:
        Sorted list of config folder names
    """
    configs_dir = get_configs_dir()
    return sorted([
        d.name for d in configs_dir.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    ])


def find_active_config() -> str | None:
    """Find config with active run (status='active').

    Used to detect if any SICA loop is currently running.
    Returns the first active config found. Use list_active_configs() for all.

    Returns:
        Config name with active run, or None if no active run
    """
    active = list_active_configs()
    return active[0] if active else None


def list_active_configs() -> list[str]:
    """List all configs with active runs (status='active').

    Returns:
        List of config names with active runs
    """
    import json

    active = []
    for name in list_configs():
        state_file = get_state_file(name)
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                if data.get("status") == "active":
                    active.append(name)
            except (json.JSONDecodeError, OSError):
                pass
    return active


def find_config_with_state(name: str | None = None) -> str | None:
    """Find config with any state (active or complete).

    Args:
        name: Specific config name to check, or None to find any

    Returns:
        Config name with state.json, or None if not found
    """
    if name:
        if get_state_file(name).exists():
            return name
        return None

    for config_name in list_configs():
        if get_state_file(config_name).exists():
            return config_name
    return None
