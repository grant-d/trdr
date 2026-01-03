"""SICA debug logging.

Enable with SICA_DEBUG=1 environment variable.
Logs to .sica/debug.log in current working directory.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

_debug_enabled = os.environ.get("SICA_DEBUG") == "1"
_debug_file = Path(".sica/debug.log") if _debug_enabled else None


def dbg(msg: str = "", reset: bool = False) -> None:
    """Write debug message to .sica/debug.log if SICA_DEBUG=1.

    Args:
        msg: Message to log. Empty string writes blank line (no timestamp).
        reset: If True, overwrite file instead of appending
    """
    if _debug_file:
        _debug_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if reset else "a"
        with open(_debug_file, mode) as f:
            if msg:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                f.write(f"[{ts}] {msg}\n")
            else:
                f.write("\n")


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return _debug_enabled
