---
description: "Show SICA loop status"
---

# SICA Status

```bash
cat .sica/current_run.json 2>/dev/null || echo "No active SICA loop"
```
