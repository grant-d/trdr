#!/usr/bin/env python3
"""Clear SICA state and stop loop.

Deletes state.json. Run archives in runs/ are preserved.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from paths import find_active_config, find_config_with_state, get_state_file, list_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear SICA state")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config name to clear (default: find active)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Clear all configs with state",
    )
    args = parser.parse_args()

    if args.all:
        # Clear all configs with state
        cleared = []
        for name in list_configs():
            state_file = get_state_file(name)
            if state_file.exists():
                state_file.unlink()
                cleared.append(name)
        if cleared:
            print(f"SICA state cleared for: {', '.join(cleared)}")
        else:
            print("No SICA states to clear")
        return

    config_name = args.config_name

    # Find config to clear
    if not config_name:
        # First try active, then any with state
        config_name = find_active_config() or find_config_with_state()

    if not config_name:
        print("No active SICA loop to clear")
        return

    state_file = get_state_file(config_name)
    if state_file.exists():
        state_file.unlink()
        print(f"SICA state cleared for config: {config_name}")
    else:
        print(f"No state found for config: {config_name}")


if __name__ == "__main__":
    main()
