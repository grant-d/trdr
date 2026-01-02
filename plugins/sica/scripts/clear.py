#!/usr/bin/env python3
"""Clear SICA state and stop loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from paths import find_active_config, get_state_file


def main() -> None:
    active = find_active_config()
    if active:
        state_file = get_state_file(active)
        state_file.unlink(missing_ok=True)
        print(f"SICA state cleared for config: {active}")
    else:
        print("No active SICA loop to clear")


if __name__ == "__main__":
    main()
