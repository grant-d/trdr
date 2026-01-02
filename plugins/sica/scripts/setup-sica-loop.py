#!/usr/bin/env python3
"""Setup a new SICA loop.

Creates state file and initial run directory.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="SICA Loop - Self-Improving Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Setup with pytest:
    /sica-loop "pytest -v" --max-iterations 20 --target-score 1.0

  Setup with specific test file:
    /sica-loop "pytest tests/test_api.py -v" --max-iterations 10

  Setup with yarn tests:
    /sica-loop "yarn test" --max-iterations 15 --completion-promise "ALL TESTS PASS"
        """,
    )

    parser.add_argument(
        "benchmark_cmd",
        nargs="?",
        help="Command to run for benchmarking (e.g., 'pytest -v')",
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=20,
        help="Maximum iterations before stopping (default: 20)",
    )
    parser.add_argument(
        "--target-score", "-t",
        type=float,
        default=1.0,
        help="Target score to achieve (0.0-1.0, default: 1.0 = all tests pass)",
    )
    parser.add_argument(
        "--completion-promise", "-p",
        type=str,
        default="TESTS PASSING",
        help="Promise phrase to signal completion (default: 'TESTS PASSING')",
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=300,
        help="Timeout for benchmark command in seconds (default: 300)",
    )
    parser.add_argument(
        "--help-full", "-H",
        action="store_true",
        help="Show detailed help",
    )

    args = parser.parse_args()

    if args.help_full:
        print("""
SICA Loop - Self-Improving Coding Agent
========================================

SICA implements an iterative self-improvement loop:

1. You provide a task and a benchmark command
2. Claude works on the task
3. When Claude tries to exit, SICA:
   - Runs the benchmark command
   - Archives the results
   - Analyzes failures
   - Continues with an improvement-focused prompt
4. Loop continues until:
   - Target score is reached
   - Max iterations reached
   - Completion promise is output

State is stored in .sica/ directory (gitignored).

Archive Structure:
  .sica/
  ├── current_run.json      # Active run state
  └── run_YYYYMMDD_HHMMSS/
      └── iteration_N/
          ├── benchmark.json    # Test results
          ├── stdout.txt        # Benchmark output
          ├── stderr.txt        # Benchmark errors
          ├── changes.diff      # Git diff
          └── summary.txt       # What Claude did

Tips:
- Use specific test commands for faster iteration
- Set realistic max-iterations (20 is usually enough)
- Target score of 0.8 = 80% tests passing
- Check .sica/ to see iteration history
""")
        sys.exit(0)

    if not args.benchmark_cmd:
        print("Error: benchmark_cmd is required", file=sys.stderr)
        print("Usage: /sica-loop 'pytest -v' --max-iterations 20", file=sys.stderr)
        sys.exit(1)

    # Create run directory
    sica_dir = Path(".sica")
    sica_dir.mkdir(exist_ok=True)

    # Add .gitignore if not exists
    gitignore = sica_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n")

    # Generate run ID
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = sica_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True)

    # Collect remaining args as the original prompt (if any come after --)
    original_prompt = ""

    # Create state
    state = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "benchmark_cmd": args.benchmark_cmd,
        "max_iterations": args.max_iterations,
        "target_score": args.target_score,
        "completion_promise": args.completion_promise,
        "benchmark_timeout": args.benchmark_timeout,
        "iteration": 0,
        "last_score": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "original_prompt": original_prompt,
    }

    # Save state
    state_file = sica_dir / "current_run.json"
    state_file.write_text(json.dumps(state, indent=2))

    # Output confirmation - includes auto-exit instruction (Ralph pattern)
    print(f"""
SICA Loop Activated
===================
Run ID: {run_id}
Benchmark: {args.benchmark_cmd}
Max iterations: {args.max_iterations}
Target score: {args.target_score}
Completion promise: {args.completion_promise}

## How SICA Works

After each change you make:
1. **Attempt to complete** - the Stop hook intercepts and runs the benchmark
2. Results archive to {run_dir}
3. If tests fail, you get a focused improvement prompt
4. Loop continues until target score ({args.target_score}) or max iterations ({args.max_iterations})

## CRITICAL INSTRUCTION

After making ANY code change, you MUST attempt to complete the conversation.
Do NOT wait for user input. Do NOT ask "should I continue?".
Just make changes, then signal completion. The hook handles the rest.

When you believe tests will pass, output:
<promise>{args.completion_promise}</promise>

To cancel: /cancel-sica

Now provide your task. After you make changes, immediately attempt to end the conversation.
""")


if __name__ == "__main__":
    main()
