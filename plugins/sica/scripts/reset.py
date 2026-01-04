#!/usr/bin/env python3
"""Reset SICA state and stop loop.

Clears config.state. Iteration archives in iterations/ are preserved.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaConfig
from debug import dbg
from paths import find_active_config, find_config_with_state, list_configs


def main() -> None:
    dbg()
    dbg("=== RESET SICA ===")

    parser = argparse.ArgumentParser(description="Reset SICA state")
    parser.add_argument(
        "config_name",
        nargs="?",
        help="Config name to reset (default: find active)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Reset all configs with state",
    )
    args = parser.parse_args()

    if args.all:
        # Reset all configs with state
        reset_configs = []
        for name in list_configs():
            try:
                config = SicaConfig.load(name)
                if config.state:
                    config.state = None
                    config.save()
                    reset_configs.append(name)
            except Exception:
                pass
        if reset_configs:
            print(f"SICA state reset for: {', '.join(reset_configs)}")
        else:
            print("No SICA states to reset")
        return

    config_name = args.config_name

    # Find config to reset
    if not config_name:
        # First try active, then any with state
        config_name = find_active_config() or find_config_with_state()

    if not config_name:
        print("No active SICA loop to reset")
        return

    try:
        config = SicaConfig.load(config_name)
        if config.state:
            config.state = None
            config.save()
            dbg(f"reset: reset {config_name}")
            print(f"SICA state reset for config: {config_name}")
        else:
            dbg(f"reset: no state for {config_name}")
            print(f"No state found for config: {config_name}")
    except FileNotFoundError:
        print(f"Config not found: {config_name}")


if __name__ == "__main__":
    main()
