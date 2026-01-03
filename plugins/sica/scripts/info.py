#!/usr/bin/env python3
"""Show SICA loop info."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaState
from debug import dbg
from paths import find_active_config, get_state_file, list_configs


def main() -> None:
    dbg()
    dbg("=== SICA INFO ===")
    active = find_active_config()
    dbg(f"status: active={active}")
    if active:
        state = SicaState.load(get_state_file(active))
        score = f"{state.last_score:.2f}" if state.last_score is not None else "N/A"

        # Substitute params in benchmark command
        bench_cmd = state.benchmark_cmd
        if state.params:
            for key, val in state.params.items():
                bench_cmd = bench_cmd.replace(f"{{{key}}}", str(val))

        print(f"Config: {active} ({state.status})")
        print(f"Run: {state.run_id}")
        print(f"Iteration: {state.iteration}/{state.max_iterations}")
        print(f"Score: {score}/{state.target_score}")

        # Recent scores trend
        if state.recent_scores:
            trend = " → ".join(f"{s:.2f}" for s in state.recent_scores[-5:])
            print(f"Trend: {trend}")

        print(f"Benchmark: {bench_cmd}")
        if state.prompt:
            print(f"Task: {state.prompt[:80]}")

        # Format params nicely
        if state.params:
            params_str = ", ".join(f"{k}={v}" for k, v in state.params.items())
            print(f"Params: {params_str}")

        print(f"Journal: {state.run_dir}/journal.md")
    else:
        print("No active SICA loop")
        configs = list_configs()
        if configs:
            print(f"Available configs: {configs}")


if __name__ == "__main__":
    main()
