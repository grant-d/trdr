# refactor: SICA Iteration Structure & Archive

## Overview

Restructure SICA config layout for cleaner iteration management. Consolidate all iteration artifacts, add file snapshots, use meaningful timestamps.

## Current Structure (Problems)

```text
.sica/configs/ethusd-vab/
  config.json
  runs/
    run_20260104_064608/        # Unnecessary nesting
      iteration_0/              # Numeric IDs not meaningful
        benchmark.json
        changes.diff
        stdout.txt
        summary.txt
        journal.md
      iteration_1/
      ...
```

**Issues:**

- `runs/run_<ts>/` layer adds indirection without benefit
- Numeric iteration IDs (0, 1, 2) not meaningful
- No file snapshots - can't easily rollback
- `context_files` conflates "read reference" with "files being evolved"

## Proposed Structure

```text
.sica/configs/ethusd-vab/
  config.json
  iterations/
    20260104-143052/            # Timestamp = meaningful, sortable
      benchmark.json
      changes.diff
      stdout.txt
      summary.txt
      journal.md
      snapshot/                 # NEW: files BEFORE this iteration
        strategy.py
        utils.py
    20260104-143215/
    20260104-143401/
```

## Config Changes

```json
{
  "benchmark_cmd": "python src/trdr/strategy/volume_area_breakout/sica_bench.py",

  "context_files": [
    "src/trdr/backtest/STRATEGY_API.md",
    "src/trdr/indicators/INDICATORS.md"
  ],

  "archive": {
    "files": [
      "src/trdr/strategy/volume_area_breakout/strategy.py"
    ]
  },

  "state": {
    "iteration": "20260104-143401",
    "iteration_count": 157,
    "last_score": 0.926
  }
}
```

**Semantics:**

- `context_files` = read-only reference for Claude (docs, examples)
- `archive.files` = files being evolved (snapshot before each iteration)
- `state.iteration` = current iteration timestamp
- `state.iteration_count` = total count for display

## Plugin Changes

### 1. Flatten Directory Structure

Change iteration path from:

```python
iteration_dir = config_dir / "runs" / run_id / f"iteration_{n}"
```

To:

```python
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
iteration_dir = config_dir / "iterations" / timestamp
```

### 2. Add Snapshot Before Iteration

```python
def snapshot_files(config: SicaConfig, iteration_dir: Path):
    """Archive files before Claude modifies them."""
    if not config.archive or not config.archive.files:
        return

    snapshot_dir = iteration_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for file_path in config.archive.files:
        src = Path(file_path)
        if src.exists():
            shutil.copy(src, snapshot_dir / src.name)
```

Called at start of each iteration, before Claude sees/modifies files.

### 3. Update State Tracking

```python
config.state.iteration = timestamp  # Current iteration ID
config.state.iteration_count += 1   # Running count
```

## Rollback Workflow

```bash
# List iterations (chronological)
ls .sica/configs/ethusd-vab/iterations/

# View iteration score
cat .sica/configs/ethusd-vab/iterations/20260104-143052/benchmark.json | jq .score

# Rollback to specific iteration
cp .sica/configs/ethusd-vab/iterations/20260104-143052/snapshot/* \
   src/trdr/strategy/volume_area_breakout/

# Or derive "after" state: snapshot of NEXT iteration
cp .sica/configs/ethusd-vab/iterations/20260104-143215/snapshot/* \
   src/trdr/strategy/volume_area_breakout/
```

## Migration

For existing configs with `runs/` structure:

```bash
# Flatten existing runs (optional - can leave as-is)
cd .sica/configs/ethusd-vab
mv runs/run_*/iteration_* iterations/
rmdir runs/run_* runs/

# Rename numeric to timestamp (or leave - they'll coexist)
```

New iterations will use timestamp format. Old numeric ones can coexist.

## Acceptance Criteria

- [ ] SICA plugin creates `iterations/<timestamp>/` instead of `runs/run_<ts>/iteration_<n>/`
- [ ] Config supports `archive.files` array
- [ ] Snapshot created before each iteration in `iterations/<ts>/snapshot/`
- [ ] `state.iteration` uses timestamp format
- [ ] `state.iteration_count` tracks total iterations
- [ ] Existing configs continue working (backward compatible)

## Files to Modify

| Location | Change |
| --- | --- |
| SICA plugin: iteration path logic | Flatten to `iterations/<timestamp>/` |
| SICA plugin: pre-iteration hook | Add snapshot_files() |
| SICA plugin: config schema | Add `archive.files` field |
| SICA plugin: state tracking | Use timestamp + count |
| `.sica/.gitignore` | Already ignores `**/runs/`, add `**/iterations/` |

## Benefits

1. **Meaningful IDs** - Timestamps tell you when, not just sequence
2. **Self-contained** - Everything about iteration in one folder
3. **Rollback** - Snapshots enable easy restore
4. **Simpler** - No `runs/run_<ts>/` indirection
5. **Generic** - Works for any SICA config, not trading-specific

## References

- [SICA Paper](https://arxiv.org/html/2504.15228v2) - "archive of all agent iterations"
- Current structure: `.sica/configs/ethusd-vab/runs/run_20260104_064608/`
