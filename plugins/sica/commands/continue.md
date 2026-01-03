---
description: "Continue SICA loop with more iterations"
argument-hint: "[CONFIG_NAME] [ADDITIONAL_ITERATIONS] [--force]"
model: haiku
---

# Continue SICA Loop

If `$1` is empty, ask the user which config to continue (use AskUserQuestion with available configs from `.sica/configs/`).

Then run the continue script:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/continue-loop.py" $ARGUMENTS
```

After continuing, make changes and exit. The stop hook will run the benchmark.
