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
    /sica-loop "pytest -v" -n 20 -s 1.0

  Setup with specific test file:
    /sica-loop "pytest tests/test_api.py -v" -n 10

  Setup with yarn tests:
    /sica-loop "yarn test" -n 15 -x "ALL TESTS PASS"
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
        "--target-score", "-s",
        type=float,
        default=1.0,
        help="Target score to achieve (0.0-1.0, default: 1.0 = all tests pass)",
    )
    parser.add_argument(
        "--exit-signal", "-x",
        type=str,
        default="TESTS PASSING",
        dest="completion_promise",
        help="Phrase to signal completion (default: 'TESTS PASSING')",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=300,
        dest="benchmark_timeout",
        help="Timeout for benchmark in seconds (default: 300)",
    )
    parser.add_argument(
        "--help-full", "-H",
        action="store_true",
        help="Show detailed help",
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="",
        help="Task description (preserved after compaction)",
    )
    parser.add_argument(
        "--file", "-f",
        action="append",
        dest="context_files",
        default=[],
        help="File to re-read after compaction (repeatable)",
    )
    parser.add_argument(
        "--journal", "-j",
        type=str,
        default="",
        help="Adopt journal from previous run (path or 'latest')",
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

    # Prompt for missing optional args (only if interactive)
    if sys.stdin.isatty():
        if not args.prompt:
            resp = input("Task description (enter to skip): ").strip()
            if resp:
                args.prompt = resp

        if not args.context_files:
            print("Context files (enter to skip, blank line to finish):")
            while True:
                resp = input("  file: ").strip()
                if not resp:
                    break
                args.context_files.append(resp)

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

    # Import journal if specified
    if args.journal:
        src = None
        if args.journal == "latest":
            runs = sorted(sica_dir.glob("run_*"), reverse=True)
            for run in runs:
                if run == run_dir:
                    continue  # Skip current run
                j = run / "journal.md"
                if j.exists():
                    src = j
                    break
            if not src:
                print("Warning: No previous journal found", file=sys.stderr)
        else:
            src = Path(args.journal)
            if not src.exists():
                print(f"Warning: Journal not found: {src}", file=sys.stderr)
                src = None

        if src:
            dest = run_dir / "journal.md"
            content = f"# Imported from {src}\n\n{src.read_text()}"
            dest.write_text(content)
            print(f"Imported journal from: {src}")

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
        "original_prompt": args.prompt,
        "context_files": args.context_files,
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

## CRITICAL - READ CAREFULLY

### EXIT AFTER EVERY CHANGE
After EACH code change, you MUST immediately attempt to end/complete the conversation.
DO NOT make multiple changes. DO NOT keep working. DO NOT ask questions.
Change → Exit → Hook runs benchmark → You get results → Repeat.

If you don't exit, the benchmark NEVER runs and iteration NEVER advances.

### OTHER RULES
- MUST use concise TI (telegraph imperative) language. Save tokens.
- DO NOT run tests manually. Hook runs benchmark on exit.
- DO NOT modify test files. Only modify source code.
- MUST maintain journal.md in {run_dir}:
  - Read FIRST before making changes
  - Log each approach (1-2 lines): what you tried, result
  - Avoid repeating failed approaches

When you believe tests will pass, output:
<promise>{args.completion_promise}</promise>

To cancel: /sica-clear

Now provide your task. After you make changes, immediately attempt to end the conversation.
""")


if __name__ == "__main__":
    main()
