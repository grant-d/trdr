#!/usr/bin/env python3
"""Continue a completed SICA loop with additional iterations."""

import argparse
import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaState
from paths import (
    find_active_config,
    get_config_dir,
    get_runs_dir,
    get_state_file,
    list_configs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue SICA loop")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config name to continue",
    )
    parser.add_argument(
        "additional",
        type=int,
        nargs="?",
        default=10,
        help="Additional iterations (default: 10)",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force continue even if active run exists",
    )
    args = parser.parse_args()

    config_name = args.config_name

    # If no config name, try to find one with a completed run
    if not config_name:
        # First check for active config
        active = find_active_config()
        if active:
            config_name = active
        else:
            # Find config with most recent final_state.json
            configs = list_configs()
            latest_config = None
            latest_time = None

            for name in configs:
                runs_dir = get_runs_dir(name)
                runs = sorted(runs_dir.glob("run_*"), reverse=True)
                for run in runs:
                    final = run / "final_state.json"
                    if final.exists():
                        mtime = final.stat().st_mtime
                        if latest_time is None or mtime > latest_time:
                            latest_time = mtime
                            latest_config = name
                        break

            if latest_config:
                config_name = latest_config

    if not config_name:
        print("Error: No config specified and no completed runs found", file=sys.stderr)
        print("Usage: /sica:sica-continue <config_name> [additional_iterations]", file=sys.stderr)
        sys.exit(1)

    # Check for active run
    state_file = get_state_file(config_name)
    if state_file.exists() and not args.force:
        print(f"Active run exists for '{config_name}'.", file=sys.stderr)
        print("Use --force to override or /sica:sica-clear first.", file=sys.stderr)
        sys.exit(1)

    # Find latest run directory with final_state.json
    runs_dir = get_runs_dir(config_name)
    runs = sorted(runs_dir.glob("run_*"), reverse=True)

    latest_run = None
    for run in runs:
        if (run / "final_state.json").exists():
            latest_run = run
            break

    if not latest_run:
        print(f"No completed runs found for '{config_name}'", file=sys.stderr)
        print("Run may still be active or was cancelled.", file=sys.stderr)
        sys.exit(1)

    final_state_file = latest_run / "final_state.json"

    # Load final state
    try:
        data = json.loads(final_state_file.read_text())
        # Handle both old format (original_prompt) and new format (prompt)
        if "original_prompt" in data and "prompt" not in data:
            data["prompt"] = data.pop("original_prompt")
        state = SicaState(
            config_name=data.get("config_name", config_name),
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
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error loading state: {e}", file=sys.stderr)
        sys.exit(1)

    # Update max iterations
    old_max = state.max_iterations
    state.max_iterations = old_max + args.additional

    # Save as active state
    state.save(state_file)

    # Build re-read instructions
    run_dir = state.run_dir
    context_files = state.context_files or []

    parts = [
        "SICA Loop Continued",
        "===================",
        f"Config: {config_name}",
        f"Resumed from: {latest_run.name}",
        f"Iteration: {state.iteration}/{state.max_iterations} (was /{old_max})",
        f"Last score: {state.last_score if state.last_score is not None else 'N/A'}",
        f"Benchmark: {state.benchmark_cmd}",
        "",
        "## RE-READ NOW",
        f"1. Read {run_dir}/journal.md",
    ]

    for i, f in enumerate(context_files, 2):
        parts.append(f"{i}. Read {f}")

    if state.prompt:
        parts.extend(["", "## Task", state.prompt])

    parts.extend([
        "",
        "Then continue improving. Stop hook runs benchmark on exit.",
    ])

    print("\n".join(parts))


if __name__ == "__main__":
    main()
