---
description: "Start SICA self-improving coding loop"
argument-hint: "CONFIG_NAME [-f] [-l]"
model: haiku
---

# SICA Loop Command

If `$1` is empty, ask the user which config to use (use AskUserQuestion with available configs from `.sica/configs/`).

Then run the setup script:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/setup-sica-loop.py" $ARGUMENTS
```

After setup completes, work on your task. When you try to exit, SICA will:

1. Run the benchmark command
2. Analyze failures
3. Archive results
4. Continue with an improvement-focused prompt if tests fail

CRITICAL: Only output the completion promise when the benchmark actually passes.
