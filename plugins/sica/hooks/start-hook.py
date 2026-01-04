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

from config import SicaConfig
from debug import dbg
from paths import find_active_config


def log(msg: str) -> None:
    """Log to stderr (visible to user)."""
    print(msg, file=sys.stderr)


def main() -> None:
    # Fast exit: no active config = not a SICA session
    config_name = find_active_config()
    if not config_name:
        sys.exit(0)

    # Read hook input
    # https://code.claude.com/docs/en/hooks#sessionstart
    # https://code.claude.com/docs/en/hooks#sessionstart-input
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        hook_input = {}

    hook_event_name = hook_input.get("hook_event_name", "")
    source = hook_input.get("source", "")
    session_id = hook_input.get("session_id", "")
    permission_mode = hook_input.get("permission_mode", "")
    cwd = hook_input.get("cwd", "")
    transcript_path = hook_input.get("transcript_path", "")
    dbg(f"hook event: {hook_event_name}, source: {source}, session_id: {session_id}, permission_mode: {permission_mode}, cwd: {cwd}, transcript_path: {transcript_path}")

    # Also constrained via "matcher" in hooks.json. Only act on `compact`
    if hook_event_name != "SessionStart" or source != "compact":
        sys.exit(0)

    # Now we know it's a SICA compaction - start logging
    dbg()
    dbg("=== SESSION START HOOK ===")
    dbg(f"Resuming {config_name}")

    try:
        config = SicaConfig.load(config_name)
    except (json.JSONDecodeError, OSError, FileNotFoundError, ValueError):
        sys.exit(0)

    state = config.state
    if not state:
        sys.exit(0)

    log("SICA: Resuming after compaction...")

    # Build context to re-inject
    context_files = config.context_files or []
    score_str = f"{state.last_score:.3f}" if state.last_score is not None else "N/A"
    run_dir = state.run_dir
    effective_max = config.max_iterations + state.iterations_added

    parts = ["SICA LOOP RESUMED AFTER COMPACTION", ""]

    if state.interpolated_prompt:
        parts.append(f"Original task: {state.interpolated_prompt}")
        parts.append("")

    if context_files:
        parts.append("IMPORTANT: Use the Read tool to re-read these files NOW:")
        for f in context_files:
            parts.append(f"- {f}")
        parts.append("")

    parts.append(f"Config: {config_name}")
    parts.append(f"Iter {state.iteration}/{effective_max} | Score {score_str}/{config.target_score}")
    parts.append(f"Benchmark: {config.benchmark_cmd}")
    parts.append("")
    parts.append("## CRITICAL - NEVER RUN BENCHMARK/TEST YOURSELF")
    parts.append("After code change, you MUST IMMEDIATELY end your turn.")
    parts.append("NEVER run benchmark/tests - hook auto-runs on exit.")
    parts.append("Change -> Exit -> Hook auto-benchmarks -> Repeat.")
    parts.append("")
    parts.append("## RULES")
    parts.append("- ONE fix per iteration, then EXIT")
    parts.append("- NEVER run benchmark yourself. Hook does it.")
    parts.append("- NO test file changes. Only modify source code.")
    parts.append(f"- MUST update {run_dir}/journal.md (BEFORE: plan, AFTER: results)")
    parts.append(f"- Done: <promise>{config.completion_promise}</promise>")

    additional_context = "\n".join(parts)

    # Output JSON for Claude Code
    # https://code.claude.com/docs/en/hooks#sessionstart-decision-control
    # https://code.claude.com/docs/en/hooks#common-json-fields
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    }))


if __name__ == "__main__":
    main()
