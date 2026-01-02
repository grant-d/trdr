#!/usr/bin/env python3

"""SICA SessionStart Hook - Re-injects context after compaction.

When session resumes after compaction, this hook:
1. Checks if SICA loop is active
2. Outputs original task + context files + iteration state
3. Instructs Claude to continue the loop
"""

import json
import sys
from pathlib import Path


def log(msg: str) -> None:
    """Log to stderr (visible to user)."""
    print(msg, file=sys.stderr)


def read_state_file() -> dict | None:
    """Read the SICA state file if it exists."""
    state_file = Path(".sica/current_run.json")
    if not state_file.exists():
        return None
    try:
        return json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def main():
    # Read hook input from stdin
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        hook_input = {}

    hook_event = hook_input.get("hook_event_name", "")
    source = hook_input.get("source", "")

    # Only act on compaction
    if hook_event != "SessionStart" or source != "compact":
        sys.exit(0)

    # Check if SICA loop is active
    state = read_state_file()
    if not state:
        sys.exit(0)

    log("SICA: Resuming after compaction...")

    # Build context to re-inject
    original_prompt = state.get("original_prompt", "")
    context_files = state.get("context_files") or []
    if not isinstance(context_files, list):
        context_files = []
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 20)
    last_score = state.get("last_score")
    target_score = state.get("target_score", 1.0)
    benchmark_cmd = state.get("benchmark_cmd", "")
    completion_promise = state.get("completion_promise", "TESTS PASSING")

    # Format context
    parts = ["SICA LOOP RESUMED AFTER COMPACTION", ""]

    if original_prompt:
        parts.append(f"Original task: {original_prompt}")
        parts.append("")

    if context_files:
        parts.append("IMPORTANT: Use the Read tool to re-read these files NOW:")
        for f in context_files:
            parts.append(f"- {f}")
        parts.append("")

    score_str = f"{last_score:.2f}" if last_score is not None else "N/A"
    parts.append(f"Current state: Iteration {iteration}/{max_iterations}, Score {score_str}/{target_score}")
    parts.append(f"Benchmark: {benchmark_cmd}")
    parts.append("")
    parts.append("## CRITICAL - EXIT AFTER EVERY CHANGE")
    parts.append("After your code change, IMMEDIATELY attempt to end/complete the conversation.")
    parts.append("DO NOT make multiple changes. DO NOT keep working. DO NOT ask questions.")
    parts.append("Change → Exit → Hook runs benchmark → You get results → Repeat.")
    parts.append("If you don't exit, the benchmark NEVER runs and iteration NEVER advances.")
    parts.append("")
    parts.append("## OTHER RULES")
    parts.append("- Concise TI (telegraph imperative) language. Save tokens.")
    parts.append("- DO NOT run tests manually. Hook runs benchmark on exit.")
    parts.append("- DO NOT modify test files. Only modify source code.")
    parts.append(f"- MUST read {state.get('run_dir', '.sica')}/journal.md FIRST - contains previous attempts.")
    parts.append("- MUST update journal.md with your approach before making changes.")
    parts.append(f"When done: <promise>{completion_promise}</promise>")

    additional_context = "\n".join(parts)

    # Output JSON for Claude Code
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    }))


if __name__ == "__main__":
    main()
