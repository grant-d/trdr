# Communication

## Style

Telegraph imperative voice. Concise, clear.
No euphoria or unnecessary validation.
Never proactively create docs unless requested.
Never write work summaries unless explicitly requested.
Study existing architecture. Match conventions.

## Documentation

### Voice

- Active voice, present tense: "Returns X" not "Will return X"
- First line: one-sentence summary (shows in tooltips)
- Be specific: list actual options, not "various options"

### TypeScript (JSDoc)

```typescript
/**
 * Brief one-line description.
 *
 * Optional detail if method is complex.
 *
 * @param name - Description with type info
 * @param config - Configuration object
 * @returns What the method returns
 * @internal   // For internal methods
 */
```

### Python (Docstrings)

```python
def function_name(param: str, config: Config) -> Result:
    """Brief one-line description.

    Optional detail if method is complex.

    Args:
        param: Description with type info
        config: Configuration object

    Returns:
        What the function returns

    Raises:
        ValueError: When input is invalid
    """
```

### @example Usage

- **Public API**: Avoid multi-line examples (hard to verify correctness)
- **Public API**: One-liner OK if needed
- **Internal code**: Full examples acceptable
