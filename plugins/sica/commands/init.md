---
description: "Initialize new SICA config interactively"
argument-hint: "CONFIG_NAME"
model: haiku
---

# Initialize SICA Config

Create a new SICA config at `.sica/configs/$1/config.json`.

If `$1` is empty, ask the user for a config name.

Then ask the user for each config field (one question at a time, use AskUserQuestion tool):

1. **benchmark_cmd** (required): Command to run for benchmarking
   - Use a `sica_bench.py` script (not pytest) to pick up code changes between iterations
   - Example: `SYMBOL={symbol} TIMEFRAME={timeframe} python path/to/sica_bench.py`
   - Use `{key}` syntax as env vars (`VAR={key} cmd`) or args (`--flag={key}`)

2. **max_iterations** (default: 20): Max improvement attempts

3. **target_score** (default: 1.0): Target score 0.0-1.0

4. **prompt** (optional): Task description
   - Example: `Optimize {symbol} strategy for {timeframe}`
   - Use `{key}` syntax for param interpolation

5. **params** (optional): Key-value pairs for interpolation
   - Example: `symbol=BTC/USD, timeframe=1h`

6. **context_files** (optional): Docs to re-read after compaction
   - Example: `docs/strategy-spec.md, research/notes.md`

After gathering answers, create the config.json file with ALL fields (use defaults for unanswered):

```json
{
  "benchmark_cmd": "<user value>",
  "max_iterations": <user value or 20>,
  "target_score": <user value or 1.0>,
  "benchmark_timeout": <user value or 120>,
  "completion_promise": "TESTS PASSING",
  "prompt": "<user value or empty string>",
  "context_files": [<user values or empty array>],
  "params": {<user key-value pairs or empty object>}
}
```

Also ensure `.sica/.gitignore` exists with content `**/runs/`.

Finally, tell the user: `Config created. Start with: /sica:loop <config_name>`
