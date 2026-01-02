"""SICA configuration and state management.

This module provides dataclasses for managing SICA loop configuration and runtime state.

Classes:
    SicaConfig: Static configuration loaded from config.json
    SicaState: Runtime state stored in state.json during active runs

Directory Structure:
    .sica/configs/<name>/
        config.json   - SicaConfig (user-editable, tracked in git)
        state.json    - SicaState (runtime, tracked in git)
        runs/         - Run archives (gitignored)

Example config.json:
    {
        "benchmark_cmd": "SYMBOL={symbol} python path/to/sica_bench.py",
        "max_iterations": 20,
        "target_score": 1.0,
        "prompt": "Optimize {symbol} strategy",
        "params": {"symbol": "BTC/USD"}
    }

Note: Use sica_bench.py (not pytest) to pick up code changes between iterations.
"""

import json
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SicaConfig:
    """SICA loop configuration loaded from config.json.

    Attributes:
        benchmark_cmd: Shell command to run for benchmarking (required)
        max_iterations: Maximum improvement attempts before stopping
        target_score: Score threshold (0.0-1.0) to consider success
        completion_promise: Phrase Claude outputs when done
        benchmark_timeout: Max seconds to wait for benchmark
        prompt: Task description with {param} placeholders
        context_files: Files to re-read after context compaction
        params: Key-value pairs for {key} interpolation in prompt and benchmark_cmd

    Example:
        >>> config = SicaConfig.load(Path(".sica/configs/btc-1h/config.json"))
        >>> config.benchmark_cmd
        'SYMBOL={symbol} python path/to/sica_bench.py'
        >>> config.interpolate_prompt()
        'Optimize BTC/USD strategy'
    """

    benchmark_cmd: str
    max_iterations: int = 20
    target_score: float = 1.0
    completion_promise: str = "TESTS PASSING"
    benchmark_timeout: int = 300
    prompt: str = ""
    context_files: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Path) -> "SicaConfig":
        """Load config from JSON file.

        Args:
            config_path: Path to config.json file

        Returns:
            SicaConfig instance with all fields populated

        Raises:
            FileNotFoundError: Config file doesn't exist
            json.JSONDecodeError: Invalid JSON syntax
            ValueError: Missing required 'benchmark_cmd' field
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        data = json.loads(config_path.read_text())

        if "benchmark_cmd" not in data:
            raise ValueError("Config requires 'benchmark_cmd'")

        return cls(
            benchmark_cmd=data["benchmark_cmd"],
            max_iterations=data.get("max_iterations", 20),
            target_score=data.get("target_score", 1.0),
            completion_promise=data.get("completion_promise", "TESTS PASSING"),
            benchmark_timeout=data.get("benchmark_timeout", 300),
            prompt=data.get("prompt", ""),
            context_files=data.get("context_files", []),
            params=data.get("params", {}),
        )

    def interpolate(self, template: str, shell_escape: bool = False) -> str:
        """Replace {key} placeholders with param values.

        Args:
            template: String with {key} placeholders
            shell_escape: If True, escape values for shell safety

        Returns:
            String with placeholders replaced by param values
        """
        if not template or not self.params:
            return template
        result = template
        for key, value in self.params.items():
            val = shlex.quote(str(value)) if shell_escape else str(value)
            result = result.replace(f"{{{key}}}", val)
        return result

    def interpolate_prompt(self) -> str:
        """Replace {key} placeholders in prompt with param values."""
        return self.interpolate(self.prompt)


@dataclass
class SicaState:
    """SICA runtime state stored in state.json during active runs.

    Created when a loop starts, updated after each iteration, deleted when
    loop completes or is cancelled. Contains both runtime data (iteration,
    scores) and a copy of config values for hook access.

    Attributes:
        config_name: Name of the config folder (e.g., 'btc-1h')
        run_id: Unique run identifier (YYYYMMDD_HHMMSS format)
        run_dir: Path to run archive directory
        iteration: Current iteration number (0-indexed)
        last_score: Most recent benchmark score (0.0-1.0)
        recent_scores: Last 10 scores for convergence detection
        started_at: ISO timestamp when run started
        benchmark_cmd: Copy from config for hook access
        max_iterations: Copy from config for hook access
        target_score: Copy from config for hook access
        completion_promise: Copy from config for hook access
        benchmark_timeout: Copy from config for hook access
        prompt: Interpolated prompt (params already substituted)
        context_files: Copy from config for hook access
        params: Copy from config for {key} interpolation

    Example:
        >>> state = SicaState.load(Path(".sica/configs/btc-1h/state.json"))
        >>> state.iteration
        5
        >>> state.last_score
        0.85
    """

    config_name: str
    run_id: str
    run_dir: str
    iteration: int = 0
    last_score: float | None = None
    recent_scores: list[float] = field(default_factory=list)
    started_at: str = ""

    # Config values copied for hook access (hooks don't load config.json)
    benchmark_cmd: str = ""
    max_iterations: int = 20
    target_score: float = 1.0
    completion_promise: str = "TESTS PASSING"
    benchmark_timeout: int = 300
    prompt: str = ""
    context_files: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls, config_name: str, run_id: str, run_dir: Path, config: SicaConfig
    ) -> "SicaState":
        """Create new state from config at start of loop.

        Copies all config values into state so hooks can access them
        without needing to load config.json separately.

        Args:
            config_name: Name of config folder (e.g., 'btc-1h')
            run_id: Unique run ID (typically YYYYMMDD_HHMMSS)
            run_dir: Path to run archive directory
            config: Loaded SicaConfig to copy values from

        Returns:
            New SicaState ready for first iteration
        """
        return cls(
            config_name=config_name,
            run_id=run_id,
            run_dir=str(run_dir),
            iteration=0,
            last_score=None,
            recent_scores=[],
            started_at=datetime.now(timezone.utc).isoformat(),
            benchmark_cmd=config.benchmark_cmd,
            max_iterations=config.max_iterations,
            target_score=config.target_score,
            completion_promise=config.completion_promise,
            benchmark_timeout=config.benchmark_timeout,
            prompt=config.interpolate_prompt(),
            context_files=list(config.context_files),
            params=dict(config.params),
        )

    @classmethod
    def load(cls, state_path: Path) -> "SicaState":
        """Load state from JSON file.

        Args:
            state_path: Path to state.json file

        Returns:
            SicaState instance with all fields populated

        Raises:
            FileNotFoundError: State file doesn't exist
            json.JSONDecodeError: Invalid JSON syntax
            ValueError: Missing required fields (config_name, run_id, run_dir)
        """
        if not state_path.exists():
            raise FileNotFoundError(f"State not found: {state_path}")

        data = json.loads(state_path.read_text())

        # Validate required fields
        for required in ("config_name", "run_id", "run_dir"):
            if required not in data:
                raise ValueError(f"State missing required field: {required}")

        return cls(
            config_name=data["config_name"],
            run_id=data["run_id"],
            run_dir=data["run_dir"],
            iteration=data.get("iteration", 0),
            last_score=data.get("last_score"),
            recent_scores=data.get("recent_scores", []),
            started_at=data.get("started_at", ""),
            benchmark_cmd=data.get("benchmark_cmd", ""),
            max_iterations=data.get("max_iterations", 20),
            target_score=data.get("target_score", 1.0),
            completion_promise=data.get("completion_promise", "TESTS PASSING"),
            benchmark_timeout=data.get("benchmark_timeout", 300),
            prompt=data.get("prompt", ""),
            context_files=data.get("context_files", []),
            params=data.get("params", {}),
        )

    def save(self, state_path: Path) -> None:
        """Save state to JSON file.

        Creates parent directories if needed. Overwrites existing file.

        Args:
            state_path: Path to state.json file
        """
        data = {
            "config_name": self.config_name,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "iteration": self.iteration,
            "last_score": self.last_score,
            "recent_scores": self.recent_scores,
            "started_at": self.started_at,
            "benchmark_cmd": self.benchmark_cmd,
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "completion_promise": self.completion_promise,
            "benchmark_timeout": self.benchmark_timeout,
            "prompt": self.prompt,
            "context_files": self.context_files,
            "params": self.params,
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(data, indent=2))

    def interpolate(self, template: str, shell_escape: bool = False) -> str:
        """Replace {key} placeholders with param values.

        Args:
            template: String with {key} placeholders
            shell_escape: If True, escape values for shell safety

        Returns:
            String with placeholders replaced by param values
        """
        if not template or not self.params:
            return template
        result = template
        for key, value in self.params.items():
            val = shlex.quote(str(value)) if shell_escape else str(value)
            result = result.replace(f"{{{key}}}", val)
        return result

    def to_dict(self) -> dict[str, str | int | float | list[str] | list[float] | dict[str, str] | None]:
        """Convert state to dict for JSON serialization.

        Includes 'original_prompt' alias for backward compatibility
        with hooks that expect that field name.

        Returns:
            Dict representation of all state fields
        """
        return {
            "config_name": self.config_name,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "iteration": self.iteration,
            "last_score": self.last_score,
            "recent_scores": self.recent_scores,
            "started_at": self.started_at,
            "benchmark_cmd": self.benchmark_cmd,
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "completion_promise": self.completion_promise,
            "benchmark_timeout": self.benchmark_timeout,
            "original_prompt": self.prompt,  # Alias for hook compat
            "context_files": self.context_files,
            "params": self.params,
        }
