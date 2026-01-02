---
description: "Show SICA loop status"
model: haiku
---

# SICA Status

```bash
if [ -f .sica/current_run.json ]; then
  python3 -c "import json; d=json.load(open('.sica/current_run.json')); print(f'Run: {d[\"run_id\"]}\nIteration: {d[\"iteration\"]}/{d[\"max_iterations\"]}\nScore: {d.get(\"last_score\", \"N/A\")}/{d[\"target_score\"]}\nBenchmark: {d[\"benchmark_cmd\"]}\nPrompt: {d.get(\"original_prompt\", \"(none)\")[:80]}')"
else
  echo "No active SICA loop"
fi
```
