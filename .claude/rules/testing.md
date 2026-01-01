# Testing

## General

- Unit + integration tests for public APIs
- Mock external dependencies
- Coverage minimum: 80%
- "units" = unit tests (e.g., "write units" means unit tests, NOT measurement)

## TypeScript

```bash
yarn test   # All tests
yarn tc     # Type check
yarn cc     # tc + lint + test
yarn lint:fix
yarn workspace [pkg] test:debug
node --import tsx --test path/file.ts
```

Troubleshooting:

- `yarn clean:tsbi` (quick) or `yarn clean` (full) then `yarn build`
- `rm -rf node_modules && yarn`, check yarn.lock conflicts

## Python

```bash
python -m pytest              # All tests
python -m pytest -v           # Verbose
python -m pytest path/test.py # Single file
python -m pytest -k "name"    # Filter by name
flake8 .                      # Lint check
```

Troubleshooting:

- Dependencies: `pip install -r requirements.txt`
- Virtual env: `python -m venv venv && source venv/bin/activate`
