# SICA - Self-Improving Coding Agent

A Claude Code plugin that implements iterative self-improvement loops. Based on the
[SICA paper](https://arxiv.org/html/2504.15228v2) and inspired by the
[ralph-wiggum](https://github.com/anthropics/claude-code/tree/HEAD/plugins/ralph-wiggum) plugin.

## How It Works

SICA implements an iterative improvement loop:

1. **Setup**: You provide a benchmark command (e.g., `pytest -v`)
2. **Work**: Claude works on your task
3. **Evaluate**: When Claude tries to exit, SICA:
   - Runs the benchmark command
   - Parses test results (pass/fail/error counts)
   - Archives results with git diffs
   - If tests fail, continues with improvement-focused prompt
4. **Loop**: Continues until target score or max iterations

## Usage

```bash
/sica-loop "pytest -v" --max-iterations 20 --target-score 1.0
```

### Arguments

- `benchmark_cmd`: Command to run for evaluation (required)
- `--max-iterations, -n`: Improvement attempts after initial benchmark (default: 20, 0 = benchmark only)
- `--target-score, -t`: Target score 0.0-1.0 (default: 1.0 = all tests pass)
- `--completion-promise, -p`: Promise phrase to signal completion (default: "TESTS PASSING")
- `--benchmark-timeout`: Timeout for benchmark in seconds (default: 300)

### Examples

```bash
# Run pytest with verbose output
/sica-loop "pytest -v"

# Target 80% passing with max 10 iterations
/sica-loop "pytest tests/test_api.py -v" --max-iterations 10 --target-score 0.8

# Use yarn for JS projects
/sica-loop "yarn test" --completion-promise "ALL SPECS GREEN"
```

### Checking Status

```bash
/sica-status
```

Shows current iteration, score, run ID, and settings.

### Stopping the Loop

**From any session:**

```bash
/sica-clear
```

**Mid-loop escape:** Output `<sica:cancel>` or `<sica:stop>` anywhere in conversation.

## Archive Structure

Results are saved in `.sica/` (gitignored):

```bash
.sica/
├── current_run.json          # Active loop state
└── run_YYYYMMDD_HHMMSS/
    └── iteration_N/
        ├── benchmark.json    # Test results, score, timing
        ├── stdout.txt        # Benchmark output
        ├── stderr.txt        # Benchmark errors
        ├── changes.diff      # Git diff of changes made
        └── summary.txt       # What Claude did
```

## Supported Test Frameworks

The benchmark parser supports:

- **pytest** (Python)
- **Jest** (JavaScript/TypeScript)
- **Mocha** (JavaScript)
- **Go test**
- **Cargo test** (Rust)

For other frameworks, SICA falls back to exit code (0 = pass, non-zero = fail).

## How It Differs from ralph-wiggum

| Feature | ralph-wiggum | SICA |
| --- | --- | ---- |
| Loop trigger | Same prompt every time | Dynamic improvement prompt |
| Evaluation | None (promise-based) | Runs benchmark, parses results |
| Archive | Single state file | Full iteration history |
| Focus | General iteration | Test-driven improvement |
| Score tracking | N/A | Per-iteration scores |

## Architecture

```bash
plugins/sica/
├── .claude-plugin/
│   └── plugin.json           # Plugin metadata
├── commands/
│   ├── sica-loop.md          # /sica-loop command
│   ├── sica-status.md        # /sica-status command
│   └── sica-clear.md         # /sica-clear command
├── hooks/
│   ├── hooks.json            # Hook configuration
│   └── stop-hook.py          # Main loop logic
├── scripts/
│   └── setup-sica-loop.py    # Loop initialization
└── README.md
```

## Credits

- Based on [SICA paper](https://arxiv.org/html/2504.15228v2) by Maxime Robeyns et al.
- Inspired by [ralph-wiggum](https://github.com/anthropics/claude-code/tree/HEAD/plugins/ralph-wiggum)
  from the Claude Code team
