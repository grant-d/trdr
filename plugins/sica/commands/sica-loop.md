---
description: "Start SICA self-improving coding loop"
argument-hint: "BENCHMARK_CMD [-p TASK] [-f FILE] [-n N] [-s SCORE] [-t SEC] [-x SIGNAL] [-j JOURNAL]"
model: haiku
---

# SICA Loop Command

Run the setup script to initialize the SICA loop:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/setup-sica-loop.py" $ARGUMENTS
```

After setup completes, work on your task. When you try to exit, SICA will:

1. Run the benchmark command
2. Analyze failures
3. Archive results
4. Continue with an improvement-focused prompt if tests fail

CRITICAL: Only output the completion promise when the benchmark actually passes.
