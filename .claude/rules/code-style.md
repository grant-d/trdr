# Code Style

## General

### Naming

- camelCase acronyms: `getUserId()` not `getUserID()`, `RagSystem` not `RAGSystem`
- Descriptive names, avoid abbreviations

### Errors

- Wrap external calls in try-catch/try-except
- Custom error classes extending base Error/Exception
- Never swallow errors silently

### Functions

- Pure functions where possible
- Early returns over nested conditionals

## TypeScript

Enforced by Prettier/ESLint (`yarn lint:fix`). Below require judgment:

- `readonly` on interface/class properties
- Discriminated unions for state/result types
- NEVER force coercions: `as any`, `as unknown as Type`
- Use `satisfies` over `as` when possible
- Single `options` object over positional params: `({ id, name }: Options)`
- Barrel files (index.ts) per domain
- USE barrels, NO re-exports between modules
- Use yarn (not npm)

## Python

Enforced by flake8/black. Below require judgment:

- Type hints for all function parameters and return types
- NEVER use `Any` type hint
- Use `dataclass` for data structures (not `dict`)
- Single `options` dataclass or kwargs over many positional params
- Imports ALWAYS at top of file
- Group imports: stdlib, third-party, local (separated by blank line)
- Use pip/poetry for dependencies
- Run `flake8` and `black` before commit

## Markdown

- Use fenced code blocks with language specified, `text` or `bash` if none
- Use 'compact' table style: `| H1 | h2 |`, `| --- | --- |`, `| row1 | 123 |`
