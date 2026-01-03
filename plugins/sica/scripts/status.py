#!/usr/bin/env python3
"""Show SICA loop status."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaState
from debug import dbg
from paths import find_active_config, get_state_file, list_configs


def main() -> None:
    dbg()
    dbg("=== SICA STATUS ===")
    active = find_active_config()
    dbg(f"status: active={active}")
    if active:
        state = SicaState.load(get_state_file(active))
        score = f"{state.last_score:.2f}" if state.last_score is not None else "N/A"
        print(f"Config: {active}")
        print(f"Run: {state.run_id}")
        print(f"Iteration: {state.iteration}/{state.max_iterations}")
        print(f"Score: {score}/{state.target_score}")
        print(f"Benchmark: {state.benchmark_cmd}")
        if state.prompt:
            print(f"Task: {state.prompt[:80]}")
        if state.params:
            print(f"Params: {state.params}")
    else:
        print("No active SICA loop")
        configs = list_configs()
        if configs:
            print(f"Available configs: {configs}")


if __name__ == "__main__":
    main()
