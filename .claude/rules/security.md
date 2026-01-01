# Security

## Secrets

- NEVER hard-code secrets, API keys, or credentials
- Use .env files for local development
- Never commit `.env` files (gitignored)

### TypeScript

- Access via `process.env` or config utility

### Python

- Access via `os.environ` or `python-dotenv`

## General

- Validate all user inputs
- Sanitize data before storage
- Use parameterized queries
- Implement rate limiting
- Follow OWASP guidelines
- Guard against prompt injection in LLM inputs
- Update dependencies regularly for security patches
