#!/usr/bin/env python3
"""Continue a completed SICA loop with additional iterations.

Reads from state.json (single source of truth), adds iterations,
sets status back to active.
"""

import argparse
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaState
from debug import dbg
from paths import (
    find_active_config,
    find_config_with_state,
    get_state_file,
    make_runtime_marker,
)


def main() -> None:
    dbg()
    dbg("=== CONTINUE SICA LOOP ===")

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
        "--force",
        "-f",
        action="store_true",
        help="Force continue even if run is active",
    )
    args = parser.parse_args()

    config_name = args.config_name

    # If no config name, try to find one with state
    if not config_name:
        # First check for active config
        active = find_active_config()
        if active:
            config_name = active
        else:
            # Find any config with state.json
            config_name = find_config_with_state()

    if not config_name:
        print("Error: No config specified and no runs found", file=sys.stderr)
        print("Usage: /sica:continue <config_name> [additional_iterations]", file=sys.stderr)
        sys.exit(1)

    # Load state
    state_file = get_state_file(config_name)
    if not state_file.exists():
        print(f"Error: No state found for '{config_name}'", file=sys.stderr)
        print("Run /sica:loop first to start a run.", file=sys.stderr)
        sys.exit(1)

    try:
        state = SicaState.load(state_file)
    except Exception as e:
        print(f"Error loading state: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if already active
    if state.status == "active" and not args.force:
        print(f"Run is already active for '{config_name}'.", file=sys.stderr)
        print("Use --force to add iterations anyway.", file=sys.stderr)
        sys.exit(1)

    # Update max iterations and set active
    old_max = state.max_iterations
    state.max_iterations = old_max + args.additional
    state.status = "active"
    # Preserve recent_scores for convergence detection
    # state.recent_scores = []

    # Save state
    state.save(state_file)
    dbg(f"Continued {config_name}: now {state.max_iterations} max iterations")

    # Build output
    run_dir = state.run_dir
    context_files = state.context_files or []

    # Runtime marker for session detection (checked by stop hook)
    marker = make_runtime_marker(config_name, state.run_id)

    parts = [
        "SICA Loop Continued",
        "===================",
        f"Config: {config_name}",
        f"Run: {state.run_id}",
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

    parts.extend(
        [
            "",
            "Then continue improving. Stop hook runs benchmark on exit.",
            "",
            marker,
        ]
    )

    print("\n".join(parts))


if __name__ == "__main__":
    main()
