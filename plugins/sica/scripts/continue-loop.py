#!/usr/bin/env python3
"""Continue a completed SICA loop with additional iterations.

Adds iterations to config.state, sets status back to active.
"""

import argparse
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaConfig
from debug import dbg
from paths import (
    find_active_config,
    find_config_with_state,
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

    # Load config
    try:
        config = SicaConfig.load(config_name)
    except FileNotFoundError:
        print(f"Error: Config not found for '{config_name}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if not config.state:
        print(f"Error: No state found for '{config_name}'", file=sys.stderr)
        print("Run /sica:loop first to start a run.", file=sys.stderr)
        sys.exit(1)

    state = config.state

    # Check if already active
    if state.status == "active" and not args.force:
        print(f"Run is already active for '{config_name}'.", file=sys.stderr)
        print("Use --force to add iterations anyway.", file=sys.stderr)
        sys.exit(1)

    # Calculate effective max before and after
    old_effective_max = config.max_iterations + state.iterations_added
    state.iterations_added += args.additional
    new_effective_max = config.max_iterations + state.iterations_added
    state.status = "active"

    # Save config (with updated state)
    config.save()
    dbg(f"Continued {config_name}: now {new_effective_max} max iterations")

    # Build output
    iterations_dir = state.run_dir  # Now points to iterations dir
    refs = config.refs

    # Runtime marker for session detection (checked by stop hook)
    marker = make_runtime_marker(config_name, state.run_id)

    parts = [
        "SICA Loop Continued",
        "===================",
        f"Config: {config_name}",
        f"Run: {state.run_id}",
        f"Iteration: {state.iteration}/{new_effective_max} (was /{old_effective_max})",
        f"Last score: {state.last_score if state.last_score is not None else 'N/A'}",
        f"Benchmark: {config.benchmark_cmd}",
        "",
        "## RE-READ NOW",
        f"1. Read recent journals in {iterations_dir}/*/journal.md",
    ]

    idx = 2
    if refs and refs.files:
        for f in refs.files:
            parts.append(f"{idx}. Read {f}")
            idx += 1
    if refs and refs.urls:
        for u in refs.urls:
            parts.append(f"{idx}. WebFetch {u}")
            idx += 1

    if state.interpolated_prompt:
        parts.extend(["", "## Task", state.interpolated_prompt])

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
