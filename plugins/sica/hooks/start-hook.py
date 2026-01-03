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

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaState
from debug import dbg
from paths import find_active_config, get_state_file


def log(msg: str) -> None:
    """Log to stderr (visible to user)."""
    print(msg, file=sys.stderr)


def main() -> None:
    # Fast exit: no active config = not a SICA session
    config_name = find_active_config()
    if not config_name:
        sys.exit(0)

    # Read hook input
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        hook_input = {}

    hook_event = hook_input.get("hook_event_name", "")
    source = hook_input.get("source", "")

    # Only act on compaction
    if hook_event != "SessionStart" or source != "compact":
        sys.exit(0)

    # Now we know it's a SICA compaction - start logging
    dbg()
    dbg("=== SESSION START HOOK ===")
    dbg(f"Resuming {config_name}")

    try:
        state = SicaState.load(get_state_file(config_name))
    except (json.JSONDecodeError, OSError, FileNotFoundError, ValueError):
        sys.exit(0)

    log("SICA: Resuming after compaction...")

    # Build context to re-inject
    context_files = state.context_files or []
    score_str = f"{state.last_score:.2f}" if state.last_score is not None else "N/A"
    run_dir = state.run_dir

    parts = ["SICA LOOP RESUMED AFTER COMPACTION", ""]

    if state.prompt:
        parts.append(f"Original task: {state.prompt}")
        parts.append("")

    if context_files:
        parts.append("IMPORTANT: Use the Read tool to re-read these files NOW:")
        for f in context_files:
            parts.append(f"- {f}")
        parts.append("")

    parts.append(f"Config: {config_name}")
    parts.append(f"Iter {state.iteration}/{state.max_iterations} | Score {score_str}/{state.target_score}")
    parts.append(f"Benchmark: {state.benchmark_cmd}")
    parts.append("")
    parts.append("## CRITICAL - EXIT AFTER EVERY CHANGE")
    parts.append("After your code change, IMMEDIATELY attempt to end/complete.")
    parts.append("Change -> Exit -> Hook benchmarks -> Repeat.")
    parts.append("")
    parts.append("## RULES")
    parts.append("- ONE fix per iteration, then exit")
    parts.append("- NO manual tests. Hook runs benchmark on exit.")
    parts.append("- NO test file changes. Only modify source code.")
    parts.append(f"- MUST update {run_dir}/journal.md (BEFORE: plan, AFTER: results)")
    parts.append(f"- Done: <promise>{state.completion_promise}</promise>")

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
