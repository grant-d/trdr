---
description: "Clear SICA state and stop loop"
model: haiku
---

# Clear SICA State

Clear the active SICA loop state:

```bash
rm -f .sica/current_run.json && echo "SICA state cleared"
```

Archive data is preserved in `.sica/run_*/`.
