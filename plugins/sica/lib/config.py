"""SICA configuration and state management.

This module provides dataclasses for managing SICA loop configuration and runtime state.

Classes:
    SicaConfig: Configuration loaded from config.json (includes nested state)
    SicaState: Runtime state nested within config.json

Directory Structure:
    .sica/configs/<name>/
        config.json   - SicaConfig with nested state (tracked in git)
        runs/         - Run archives (gitignored)

Example config.json:
    {
        "benchmark_cmd": "SYMBOL={symbol} python path/to/sica_bench.py",
        "max_iterations": 20,
        "target_score": 1.0,
        "prompt": "Optimize {symbol} strategy",
        "params": {"symbol": "BTC/USD"},
        "state": {
            "run_id": "20260104_005010",
            "status": "active",
            "iteration": 0,
            ...
        }
    }

Note: Use sica_bench.py (not pytest) to pick up code changes between iterations.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

ThinkLevel = Literal["Think", "Think hard", "Think harder", "Ultrathink", ""]


@dataclass
class SicaState:
    """SICA runtime state nested within config.json.

    Attributes:
        run_id: Unique run identifier (YYYYMMDD_HHMMSS format)
        run_dir: Path to run archive directory
        status: Run status ("active" | "complete")
        iteration: Current iteration number (0-indexed)
        last_score: Most recent benchmark score (0.0-1.0)
        recent_scores: All scores (convergence + LLM journal)
        started_at: ISO timestamp when run started
        completed_at: ISO timestamp when run completed (if complete)
        iterations_added: Additional iterations from /sica:continue
        interpolated_prompt: Prompt with params substituted, frozen at loop start
    """

    run_id: str
    run_dir: str
    status: str = "active"
    iteration: int = 0
    last_score: float | None = None
    recent_scores: list[float] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    iterations_added: int = 0
    interpolated_prompt: str = ""

    @classmethod
    def create(cls, run_id: str, run_dir: Path, interpolated_prompt: str) -> "SicaState":
        """Create new state at start of loop."""
        return cls(
            run_id=run_id,
            run_dir=str(run_dir),
            status="active",
            iteration=0,
            last_score=None,
            recent_scores=[],
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at="",
            iterations_added=0,
            interpolated_prompt=interpolated_prompt,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SicaState":
        """Load state from dict."""
        # Migration: old files have 'prompt', new have 'interpolated_prompt'
        interpolated_prompt = data.get("interpolated_prompt", "")
        if not interpolated_prompt:
            interpolated_prompt = data.get("prompt", "")

        return cls(
            run_id=data.get("run_id", ""),
            run_dir=data.get("run_dir", ""),
            status=data.get("status", "active"),
            iteration=data.get("iteration", 0),
            last_score=data.get("last_score"),
            recent_scores=data.get("recent_scores", []),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            iterations_added=data.get("iterations_added", 0),
            interpolated_prompt=interpolated_prompt,
        )

    def to_dict(self) -> dict:
        """Convert state to dict for JSON serialization."""
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "status": self.status,
            "iteration": self.iteration,
            "last_score": self.last_score,
            "recent_scores": self.recent_scores,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "iterations_added": self.iterations_added,
            "interpolated_prompt": self.interpolated_prompt,
        }


@dataclass
class SicaConfig:
    """SICA loop configuration loaded from config.json.

    Attributes:
        name: Config name (folder name, set on load)
        benchmark_cmd: Shell command to run for benchmarking (required)
        max_iterations: Maximum improvement attempts before stopping
        target_score: Score threshold (0.0-1.0) to consider success
        completion_promise: Phrase Claude outputs when done
        benchmark_timeout: Max seconds to wait for benchmark
        prompt: Task description with {param} placeholders
        context_files: Files to re-read after context compaction
        params: Key-value pairs for {key} interpolation in prompt and benchmark_cmd
        state: Runtime state (None if no active/complete run)

    Example:
        >>> config = SicaConfig.load("btc-1h")
        >>> config.benchmark_cmd
        'SYMBOL={symbol} python path/to/sica_bench.py'
        >>> config.state.iteration
        5
    """

    name: str
    benchmark_cmd: str
    max_iterations: int = 20
    target_score: float = 1.0
    completion_promise: str = "TESTS PASSING"
    benchmark_timeout: int = 120
    prompt: str = ""
    context_files: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)
    state: SicaState | None = None

    @classmethod
    def load(cls, config_name: str) -> "SicaConfig":
        """Load config from JSON file.

        Args:
            config_name: Name of the config folder (e.g., 'btc-1h')

        Returns:
            SicaConfig instance with all fields populated

        Raises:
            FileNotFoundError: Config file doesn't exist
            json.JSONDecodeError: Invalid JSON syntax
            ValueError: Missing required 'benchmark_cmd' field
        """
        from paths import get_config_file

        config_path = get_config_file(config_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        data = json.loads(config_path.read_text())

        if "benchmark_cmd" not in data:
            raise ValueError("Config requires 'benchmark_cmd'")

        state = None
        if "state" in data and data["state"]:
            state = SicaState.from_dict(data["state"])

        return cls(
            name=config_name,
            benchmark_cmd=data["benchmark_cmd"],
            max_iterations=data.get("max_iterations", 20),
            target_score=data.get("target_score", 1.0),
            completion_promise=data.get("completion_promise", "TESTS PASSING"),
            benchmark_timeout=data.get("benchmark_timeout", 120),
            prompt=data.get("prompt", ""),
            context_files=data.get("context_files", []),
            params=data.get("params", {}),
            state=state,
        )

    def save(self) -> None:
        """Save config to JSON file."""
        from paths import get_config_file

        data = {
            "benchmark_cmd": self.benchmark_cmd,
            "max_iterations": self.max_iterations,
            "target_score": self.target_score,
            "benchmark_timeout": self.benchmark_timeout,
            "completion_promise": self.completion_promise,
            "prompt": self.prompt,
            "context_files": self.context_files,
            "params": self.params,
            "state": self.state.to_dict() if self.state else None,
        }
        config_path = get_config_file(self.name)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(data, indent=2) + "\n")

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

    def interpolate_prompt(self, think: ThinkLevel = "") -> str:
        """Replace {key} placeholders in prompt with param values."""
        return self.interpolate(self.prompt) + ("\n" + think if think else "")
