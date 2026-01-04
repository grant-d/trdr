"""SICA path resolution utilities.

Provides functions for resolving paths within the .sica directory structure.
All paths are relative to current working directory.

Directory Structure:
    .sica/                          # Root (created on first use)
    ├── .gitignore                  # Contains: **/runs/
    └── configs/                    # Config folders
        └── <name>/                 # Individual config
            ├── config.json         # Config + nested state
            └── runs/               # Run archives (gitignored)
                └── run_YYYYMMDD_HHMMSS/
                    ├── journal.md
                    └── iteration_N/
"""

import json
from pathlib import Path


def make_runtime_marker(config_name: str, run_id: str) -> str:
    """Create SICA runtime marker for session detection.

    Embed at END of prompts (less likely to be culled by context length).
    Stop hook checks for this marker to distinguish SICA sessions.

    Args:
        config_name: Config name
        run_id: Run ID

    Returns:
        Marker string like <SICA_RT_MARKER config="name" run="id"/>
    """
    return f'<SICA_RT_MARKER config="{config_name}" run="{run_id}"/>'


def is_sica_session(transcript_path: str, config_name: str, run_id: str) -> bool:
    """Check if transcript contains matching SICA runtime marker.

    Args:
        transcript_path: Path to transcript file
        config_name: Expected config name
        run_id: Expected run ID

    Returns:
        True if marker matches
    """
    if not transcript_path or not config_name or not run_id:
        return False
    path = Path(transcript_path)
    if not path.exists():
        return False
    try:
        content = path.read_text()
        marker = make_runtime_marker(config_name, run_id)
        # Transcript is JSONL so quotes are escaped
        escaped = marker.replace('"', '\\"')
        return escaped in content
    except Exception:
        return False


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
    """List all configs with active runs (state.status='active').

    Returns:
        List of config names with active runs
    """
    active = []
    for name in list_configs():
        config_file = get_config_file(name)
        try:
            data = json.loads(config_file.read_text())
            state = data.get("state")
            if state and state.get("status") == "active":
                active.append(name)
        except (json.JSONDecodeError, OSError):
            pass
    return active


def find_config_with_state(name: str | None = None) -> str | None:
    """Find config with any state (active or complete).

    Args:
        name: Specific config name to check, or None to find any

    Returns:
        Config name with state, or None if not found
    """
    if name:
        config_file = get_config_file(name)
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text())
                if data.get("state"):
                    return name
            except (json.JSONDecodeError, OSError):
                pass
        return None

    for config_name in list_configs():
        config_file = get_config_file(config_name)
        try:
            data = json.loads(config_file.read_text())
            if data.get("state"):
                return config_name
        except (json.JSONDecodeError, OSError):
            pass
    return None
