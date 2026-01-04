# refactor: Separate SICA config from state

## Overview

Reduce duplication between `SicaConfig` and `SicaState` in the SICA plugin. Config is the canonical source of truth for loop parameters. State should only track runtime values.

## Problem Statement

Currently 8 fields are duplicated between config and state:

| Field | In Config | In State | Reason for Duplication |
| --- | --- | --- | --- |
| `benchmark_cmd` | Yes | Yes | "hooks don't load config.json" |
| `max_iterations` | Yes | Yes | Same |
| `target_score` | Yes | Yes | Same |
| `completion_promise` | Yes | Yes | Same |
| `benchmark_timeout` | Yes | Yes | Same |
| `prompt` | Yes (raw) | Yes (interpolated) | Same |
| `context_files` | Yes | Yes | Same |
| `params` | Yes | Yes | Same |

This creates:

- Knowledge duplication - same values in two places
- Sync risk - if config changes, state has stale copies
- Bloated state.json - 17 fields when 6-8 suffice
- Divergence bugs - e.g., `max_iterations` modified by `/sica:continue` but config unchanged

## Proposed Solution

### Core Changes

1. **Remove duplicated fields from `SicaState`**
2. **Create `get_loop_context()` helper** returning `(config, state)` tuple
3. **Update all hooks/scripts** to load both files
4. **Handle `max_iterations` override** via explicit `iterations_added` field

### State After Refactor

```python
@dataclass
class SicaState:
    """Runtime-only values. Mutable."""
    config_name: str
    run_id: str
    run_dir: str
    status: str = "active"  # "active" | "complete"
    iteration: int = 0
    last_score: float | None = None
    recent_scores: list[float] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""

    # Runtime overrides (not duplicates)
    iterations_added: int = 0  # From /sica:continue
    interpolated_prompt: str = ""  # Frozen at loop start
```

### Config Remains Unchanged

```python
@dataclass
class SicaConfig:
    """Canonical source of truth. Immutable after load."""
    benchmark_cmd: str
    max_iterations: int = 20
    target_score: float = 1.0
    completion_promise: str = "TESTS PASSING"
    benchmark_timeout: int = 120
    prompt: str = ""
    context_files: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)
```

### Access Pattern

```python
def get_loop_context(config_name: str) -> tuple[SicaConfig, SicaState]:
    """Load both config and state for a SICA loop."""
    config = SicaConfig.load(get_config_file(config_name))
    state = SicaState.load(get_state_file(config_name))
    return config, state

# In hooks:
config, state = get_loop_context(config_name)
effective_max = config.max_iterations + state.iterations_added
if state.iteration >= effective_max:
    # ...
```

## Technical Approach

### Phase 1: Add Helper and Prepare State

**Files:**

- `plugins/sica/lib/config.py`

**Tasks:**

1. Add `get_loop_context()` function
2. Add `iterations_added: int = 0` field to `SicaState`
3. Add `interpolated_prompt: str = ""` field to `SicaState`
4. Keep duplicated fields temporarily for backwards compat
5. Add `effective_max_iterations` property to `SicaState` that takes config

### Phase 2: Update Hooks to Use Context

**Files:**

- `plugins/sica/hooks/stop-hook.py`
- `plugins/sica/hooks/start-hook.py`

**Tasks:**

1. Replace `state = SicaState.load()` with `config, state = get_loop_context()`
2. Change all `state.benchmark_cmd` → `config.benchmark_cmd`
3. Change all `state.target_score` → `config.target_score`
4. Change `state.max_iterations` → `config.max_iterations + state.iterations_added`
5. Change `state.completion_promise` → `config.completion_promise`
6. Change `state.benchmark_timeout` → `config.benchmark_timeout`
7. Change `state.context_files` → `config.context_files`
8. Use `state.interpolated_prompt` for prompt (frozen at start)

### Phase 3: Update Scripts

**Files:**

- `plugins/sica/scripts/setup-loop.py`
- `plugins/sica/scripts/continue-loop.py`
- `plugins/sica/scripts/info.py`

**Tasks:**

1. `setup-loop.py`: Set `state.interpolated_prompt` instead of `state.prompt`
2. `setup-loop.py`: Remove copying of config fields to state
3. `continue-loop.py`: Set `state.iterations_added += additional` instead of modifying `state.max_iterations`
4. `info.py`: Use `get_loop_context()`, display from config + state appropriately

### Phase 4: Remove Deprecated Fields

**Files:**

- `plugins/sica/lib/config.py`

**Tasks:**

1. Remove `benchmark_cmd`, `max_iterations`, `target_score`, `completion_promise`, `benchmark_timeout`, `prompt`, `context_files`, `params` from `SicaState`
2. Update `SicaState.load()` to handle old state files gracefully (ignore extra fields)
3. Update `SicaState.create()` to not copy config fields

## Acceptance Criteria

### Functional Requirements

- [ ] Hooks load both config.json and state.json
- [ ] Config fields accessed from config, not state
- [ ] `/sica:continue` increments `iterations_added` not `max_iterations`
- [ ] `/sica:info` displays correct values from appropriate source
- [ ] Existing loops continue working (migration)

### Non-Functional Requirements

- [ ] state.json file size reduced (fewer fields)
- [ ] Clear separation: config = what to do, state = what happened
- [ ] Prompt frozen at loop start (current behavior preserved)

## Migration Strategy

**For existing state.json files:**

1. `SicaState.load()` gracefully ignores unknown fields (already does via `get()`)
2. If `iterations_added` missing, default to 0
3. If `interpolated_prompt` missing, fall back to `prompt` field if present
4. Old fields in state.json remain but are ignored

**No breaking changes** - old state files load successfully, new fields have sensible defaults.

## Edge Cases

| Scenario | Behavior |
| --- | --- |
| Config edited mid-run | Hooks see new config values (intentional) |
| Config deleted mid-run | Hook raises FileNotFoundError |
| Old state.json format | Loads successfully, extra fields ignored |
| `/sica:continue` on fresh run | `iterations_added` starts at 0, adds normally |

## Files to Modify

| File | Changes |
| --- | --- |
| `plugins/sica/lib/config.py:122-331` | Add `get_loop_context()`, add fields, remove deprecated |
| `plugins/sica/hooks/stop-hook.py:47,182-496` | Use context, access config fields |
| `plugins/sica/hooks/start-hook.py:59-96` | Use context, access config fields |
| `plugins/sica/scripts/setup-loop.py:118-147` | Remove field copying |
| `plugins/sica/scripts/continue-loop.py:75-94` | Use `iterations_added` |
| `plugins/sica/scripts/info.py:20-48` | Display from config + state |

## Testing

Manual verification:

1. Start new loop with `/sica:loop` - verify state.json smaller
2. Run `/sica:info` - verify correct display
3. Run `/sica:continue 10` - verify iterations extend correctly
4. Edit config.json mid-run - verify hooks see new values
5. Load old state.json - verify no errors

## References

### Internal

- `plugins/sica/lib/config.py:34-120` - SicaConfig definition
- `plugins/sica/lib/config.py:122-331` - SicaState definition
- `plugins/sica/lib/config.py:166-174` - Current duplication comment
- `plugins/sica/scripts/continue-loop.py:88` - max_iterations mutation

### Best Practices

- [Single Source of Truth](https://en.wikipedia.org/wiki/Single_source_of_truth)
- [Python Frozen Dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [ConfigState Pattern](https://pypi.org/project/config-state/)
