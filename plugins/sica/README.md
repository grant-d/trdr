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
/sica-loop "pytest -v" -n 20 -s 1.0
```

### Arguments

- `benchmark_cmd`: Command to run for evaluation (required)
- `--prompt, -p`: Task description (preserved after compaction)
- `--file, -f`: File to re-read after compaction (repeatable)
- `--max-iterations, -n`: Improvement attempts (default: 20, 0 = benchmark only)
- `--target-score, -s`: Target score 0.0-1.0 (default: 1.0)
- `--exit-signal, -x`: Phrase to signal completion (default: "TESTS PASSING")
- `--timeout, -t`: Benchmark timeout in seconds (default: 300)
- `--journal, -j`: Adopt journal from previous run (path or 'latest')

### Examples

```bash
# Run pytest with verbose output
/sica-loop "pytest -v"

# With task and context file (preserved after compaction)
/sica-loop "pytest -v" -p "Fix auth bug" -f docs/auth-spec.md

# Target 80% passing with max 10 iterations
/sica-loop "pytest tests/test_api.py -v" -n 10 -s 0.8

# Use yarn for JS projects
/sica-loop "yarn test" -x "ALL SPECS GREEN"

# Import learnings from previous session
/sica-loop "pytest -v" -j latest
```

### Checking Status

```bash
/sica-status
```

Shows current iteration, score, run ID, and settings.

### Continuing a Completed Run

```bash
/sica-continue 10    # Add 10 more iterations
/sica-continue       # Add 10 (default)
/sica-continue --force  # Override active run
```

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
    ├── journal.md            # Claude's log of approaches tried
    ├── final_state.json      # State at completion (for /sica-continue)
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
│   └── plugin.json             # Plugin metadata
├── commands/
│   ├── sica-loop.md            # /sica-loop command
│   ├── sica-status.md          # /sica-status command
│   ├── sica-clear.md           # /sica-clear command
│   └── sica-continue.md        # /sica-continue command
├── hooks/
│   ├── hooks.json              # Hook configuration
│   ├── stop-hook.py            # Main loop logic
│   └── session-start-hook.py   # Compaction recovery
├── scripts/
│   ├── setup-sica-loop.py      # Loop initialization
│   └── continue-sica-loop.py   # Continue completed run
└── README.md
```

## Development

After editing plugin files, sync to cache:

```bash
cp -r plugins/sica/* ~/.claude/plugins/cache/jigx-plugins/sica/1.0.0/
```

## Credits

- Based on [SICA paper](https://arxiv.org/html/2504.15228v2) by Maxime Robeyns et al.
- Inspired by [ralph-wiggum](https://github.com/anthropics/claude-code/tree/HEAD/plugins/ralph-wiggum)
  from the Claude Code team
