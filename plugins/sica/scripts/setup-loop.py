#!/usr/bin/env python3
"""Setup a new SICA loop.

Creates state file and run directory within config folder.
"""

import argparse
import sys
from datetime import datetime, timezone
import json
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaConfig, SicaState
from debug import dbg
from paths import (
    find_active_config,
    get_config_file,
    get_run_dir,
    get_sica_root,
    get_state_file,
    list_configs,
    make_runtime_marker,
)

# Plugin paths for dev sync
# PLUGIN_SRC = Path(__file__).parent.parent  # plugins/sica/
# PLUGIN_CACHE = Path.home() / ".claude/plugins/cache/jigx-plugins/sica/1.0.0"


# def sync_plugin_to_cache() -> None:
#     """Sync plugin source to cache for development."""
#     if not PLUGIN_SRC.exists():
#         return
#     try:
#         if PLUGIN_CACHE.exists():
#             shutil.rmtree(PLUGIN_CACHE)
#         shutil.copytree(PLUGIN_SRC, PLUGIN_CACHE)
#         dbg(f"Synced plugin to cache: {PLUGIN_CACHE}")
#     except Exception as e:
#         dbg(f"Plugin sync failed: {e}")


def main() -> None:
    # sync_plugin_to_cache()
    dbg("=== SETUP SICA LOOP ===", reset=True)

    parser = argparse.ArgumentParser(
        description="SICA Loop - Self-Improving Coding Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  /sica:loop btc-1h         # Run with config
  /sica:loop --list         # List configs
""",
    )

    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config name from .sica/configs/<name>/",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_configs",
        help="List available configs",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force start even if another config has active state",
    )

    args = parser.parse_args()

    # List configs
    if args.list_configs:
        configs = list_configs()
        if configs:
            print("Available configs:")
            for name in configs:
                state_file = get_state_file(name)
                if state_file.exists():
                    try:
                        data = json.loads(state_file.read_text())
                        status = f" ({data.get('status', 'unknown')})"
                    except (json.JSONDecodeError, OSError):
                        status = " (state error)"
                else:
                    status = ""
                print(f"  {name}{status}")
        else:
            print("No configs found. Create one at .sica/configs/<name>/config.json")
        return

    # Require config name
    if not args.config_name:
        print("Error: config_name required", file=sys.stderr)
        print("Usage: /sica:loop <config_name>", file=sys.stderr)
        print("       /sica:loop --list", file=sys.stderr)
        sys.exit(1)

    config_name = args.config_name

    # Check for active run in another config
    active = find_active_config()
    if active and active != config_name and not args.force:
        print(f"Error: Active run in '{active}'", file=sys.stderr)
        print("Use --force to override or /sica:reset to stop it", file=sys.stderr)
        sys.exit(1)

    # Load config
    config_path = get_config_file(config_name)
    try:
        config = SicaConfig.load(config_path)
    except FileNotFoundError:
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        print("Create it or use /sica:loop --list to see available configs", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate run ID and create run directory
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = get_run_dir(config_name, run_id)

    # Ensure .sica/.gitignore exists
    gitignore = get_sica_root() / ".gitignore"
    if not gitignore.exists():
        gitignore.parent.mkdir(parents=True, exist_ok=True)
        gitignore.write_text("**/runs/\ndebug.log\n")

    # Create empty journal
    journal_path = run_dir / "journal.md"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text("# SICA Journal\n\n")

    # Create state
    state = SicaState.create(config_name, run_id, run_dir, config)

    # Save state
    state_path = get_state_file(config_name)
    state.save(state_path)
    dbg(f"Created run {run_id} for {config_name}")

    # Show params if any
    params_info = ""
    if config.params:
        params_lines = [f"  {k}: {v}" for k, v in config.params.items()]
        params_info = "\nParams:\n" + "\n".join(params_lines)

    prompt = config.interpolate_prompt()

    # Runtime marker for session detection (checked by stop hook)
    marker = make_runtime_marker(config_name, run_id)

    print(f"""
SICA Loop Activated
===================
Config: {config_name}
Run ID: {run_id}
Benchmark: {config.benchmark_cmd}
Max iterations: {config.max_iterations}
Target score: {config.target_score}
Completion promise: {config.completion_promise}{params_info}

## How SICA Works

After each change:
1. **Exit** - hook intercepts and runs benchmark
2. Results archive to {run_dir}
3. If tests fail, you get improvement prompt
4. Loop until target ({config.target_score}) or max ({config.max_iterations})

## CRITICAL

### EXIT AFTER EVERY CHANGE
After EACH code change, immediately attempt to end/complete.
Change -> Exit -> Hook benchmarks -> Repeat.

### RULES
- Concise TI language. Save tokens.
- NO manual tests. Hook runs benchmark on exit.
- NO test file changes. Only source code.
- MUST maintain journal.md in {run_dir}:
  - Read FIRST before changes
  - Log each approach (1-2 lines)
  - Avoid repeating failed approaches

Done: <promise>{config.completion_promise}</promise>
Stop: Press Esc or run /sica:reset

{"Task: " + prompt if prompt else "Provide your task."} Then make changes and exit.

{marker}
""")


if __name__ == "__main__":
    main()
