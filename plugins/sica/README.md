# SICA - Self-Improving Coding Agent

A Claude Code plugin that implements iterative self-improvement loops.

## How It Works

SICA implements an iterative improvement loop:

1. **Setup**: Create a config with benchmark command and params
2. **Work**: Claude works on your task
3. **Evaluate**: When Claude tries to exit, SICA:
   - Runs the benchmark command
   - Parses test results (pass/fail/error counts)
   - Archives results with git diffs
   - If tests fail, continues with improvement-focused prompt
4. **Loop**: Continues until target score or max iterations

## Usage

### Interactive Setup

```bash
/sica:init btc-1h    # Prompts for each config property
```

### Config-Based

Create a config at `.sica/configs/<name>/config.json`:

```json
{
  "benchmark_cmd": "SYMBOL={symbol} TIMEFRAME={timeframe} python path/to/sica_bench.py",
  "max_iterations": 20,
  "target_score": 1.0,
  "prompt": "Optimize {symbol} strategy for {timeframe}",
  "context_files": ["docs/strategy-spec.md", "research/backtest-notes.md"],
  "params": {
    "symbol": "BTC/USD",
    "timeframe": "1h"
  }
}
```

Params use `{key}` syntax. Works as env vars (`VAR={key} cmd`) or args (`--flag={key}`).

**Important**: Use a dedicated `sica_bench.py` script instead of `pytest` for benchmarks.
The script must force-reload strategy modules to pick up code changes between iterations.
See `src/trdr/strategy/*/sica_bench.py` for examples.

Then run:

```bash
/sica:loop btc-1h         # Start loop with config
/sica:loop --list         # List available configs
```

### Arguments

| Arg | Description |
| --- | --- |
| `config_name` | Config folder name (required) |
| `-f, --force` | Force start if another config is active |
| `-l, --list` | List available configs |

### Config Fields

| Field | Description | Default |
| --- | --- | --- |
| `benchmark_cmd` | Command to run (required) | - |
| `max_iterations` | Max improvement attempts | 20 |
| `target_score` | Target score 0.0-1.0 | 1.0 |
| `completion_promise` | Phrase to signal completion | "TESTS PASSING" |
| `benchmark_timeout` | Timeout in seconds | 120 |
| `prompt` | Task description (supports `{param}` interpolation) | "" |
| `context_files` | Docs to re-read after compaction (specs, research notes) | [] |
| `params` | Key-value pairs for `{key}` interpolation | {} |

### Commands

```bash
/sica:init btc-1h      # Create config interactively
/sica:loop btc-1h      # Start loop with config
/sica:info             # Show current iteration, score, settings
/sica:continue btc-1h  # Add 10 more iterations to completed run
/sica:reset            # Stop active loop (or press Esc)
```

## Directory Structure

```text
.sica/                                    # Tracked in git
├── .gitignore                            # Ignores **/runs/
└── configs/
    └── btc-1h/                           # Config folder
        ├── config.json                   # Config params (tracked)
        ├── state.json                    # Run state (tracked, single source of truth)
        └── runs/                         # Archives (ignored)
            └── run_YYYYMMDD_HHMMSS/
                ├── journal.md            # Claude's log
                └── iteration_N/
                    ├── benchmark.json    # Test results, score
                    ├── stdout.txt
                    ├── stderr.txt
                    ├── changes.diff      # Git diff
                    └── summary.txt
```

## Supported Test Frameworks

The benchmark parser supports:

- **pytest** (Python)
- **Jest** (JavaScript/TypeScript)
- **Mocha** (JavaScript)
- **Go test**
- **Cargo test** (Rust)

For other frameworks, SICA falls back to exit code (0 = pass, non-zero = fail).

### Custom Benchmarks

For custom benchmark scripts (like `sica_bench.py`), MUST output pytest-style summary for proper score tracking:

```python
# At end of script
if failures:
    print(f"{len(failures)} failed, 0 passed")
    sys.exit(1)
else:
    print("1 passed, 0 failed")
    sys.exit(0)
```

This enables SICA to show `0✓ 3✗` instead of just pass/fail.

## Development

After editing plugin files, sync to cache:

```bash
cp -r plugins/sica/* ~/.claude/plugins/cache/jigx-plugins/sica/1.0.0/
```

### Debug Mode

Enable verbose logging to `.sica/debug.log`:

```bash
export SICA_DEBUG=1
```

## Resources

For learning Claude Code plugin development:

- [Official "Create plugins" docs](https://code.claude.com/docs/en/plugins) - Covers plugin structure, manifests, commands, agents, skills, hooks, MCP, local dev, and marketplaces
- [First plugin walkthrough](https://alexop.dev/posts/building-my-first-claude-code-plugin/) - End-to-end workflow with best practices

## Credits

- Based on [SICA paper](https://arxiv.org/html/2504.15228v2) by Maxime Robeyns et al.
- Inspired by [ralph-wiggum](https://github.com/anthropics/claude-code/tree/HEAD/plugins/ralph-wiggum)
  from the Claude Code team
