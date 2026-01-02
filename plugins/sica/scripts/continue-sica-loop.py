#!/usr/bin/env python3
"""Continue a completed SICA loop with additional iterations."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Continue SICA loop")
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

    sica_dir = Path(".sica")
    current_run = sica_dir / "current_run.json"

    # Check for active run
    if current_run.exists() and not args.force:
        print("Active run exists. Use --force to override or /sica-clear first.", file=sys.stderr)
        sys.exit(1)

    # Find latest run directory
    runs = sorted(sica_dir.glob("run_*"), reverse=True)
    if not runs:
        print("No previous runs found in .sica/", file=sys.stderr)
        sys.exit(1)

    latest = runs[0]
    state_file = latest / "final_state.json"

    if not state_file.exists():
        print(f"No final_state.json in {latest}", file=sys.stderr)
        print("Run may still be active or was cancelled.", file=sys.stderr)
        sys.exit(1)

    # Load and update state
    state = json.loads(state_file.read_text())
    old_max = state["max_iterations"]
    state["max_iterations"] = old_max + args.additional

    # Write to current_run.json
    current_run.write_text(json.dumps(state, indent=2))

    print(f"""
SICA Loop Continued
===================
Resumed from: {latest.name}
Iteration: {state['iteration']}/{state['max_iterations']} (was /{old_max})
Last score: {state.get('last_score', 'N/A')}
Benchmark: {state.get('benchmark_cmd', 'N/A')}

Continue working. The stop hook will run on next exit attempt.
""")


if __name__ == "__main__":
    main()
