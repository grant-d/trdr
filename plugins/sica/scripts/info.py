#!/usr/bin/env python3
"""Show SICA loop info."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "lib"))

from config import SicaConfig, SicaState, get_loop_context
from debug import dbg
from paths import find_active_config, get_state_file, list_configs


def main() -> None:
    dbg()
    dbg("=== SICA INFO ===")
    active = find_active_config()
    dbg(f"status: active={active}")
    if active:
        config, state = get_loop_context(active)
        score = f"{state.last_score:.2f}" if state.last_score is not None else "N/A"
        effective_max = config.max_iterations + state.iterations_added

        # Substitute params in benchmark command
        bench_cmd = config.interpolate(config.benchmark_cmd)

        print(f"Config: {active} ({state.status})")
        print(f"Run: {state.run_id}")
        print(f"Iteration: {state.iteration}/{effective_max}")
        print(f"Score: {score}/{config.target_score}")

        # Recent scores trend
        if state.recent_scores:
            trend = " → ".join(f"{s:.2f}" for s in state.recent_scores[-5:])
            print(f"Trend: {trend}")

        print(f"Benchmark: {bench_cmd}")
        if state.interpolated_prompt:
            print(f"Task: {state.interpolated_prompt[:80]}")

        # Format params nicely
        if config.params:
            params_str = ", ".join(f"{k}={v}" for k, v in config.params.items())
            print(f"Params: {params_str}")

        print(f"Journal: {state.run_dir}/journal.md")
    else:
        print("No active SICA loop")
        configs = list_configs()
        if configs:
            print(f"Available configs: {configs}")


if __name__ == "__main__":
    main()
