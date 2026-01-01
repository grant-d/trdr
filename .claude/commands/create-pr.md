---
model: haiku
---

# GitHub Pull Request Creator

Create well-structured pull requests for the Jigx monorepo. Telegraph-imperative voice throughout.

## Format Rules

**Telegraph-imperative voice**: "Fix generation", "Add builder", "Remove deprecated" - not "This PR fixes generation issues".

**Types**: fix, feat, perf, chore, test, docs, style, refactor, build, ci
**Scopes**: sdk, expert-sdk, mcp, examples, web, gcp, types, builders, components, datasources, screens, actions, dev-tools

**Formats**:

- PR title: `type(scope): description`
- Branch: `type/scope-short-description` (kebab, lower, abbreviate long words)
- Commit: `type(scope): description`
- ClickUp link: `Related to https://app.clickup.com/t/{taskId} CU-{taskId}`
  - **ClickUp ID formats**: Both `CU-868ezxd5u` and `https://app.clickup.com/t/868ezxd5u` identify the same task
  - Strip `CU-` prefix from task IDs when creating URLs

## Workflow

### 1. Git Check

- **Parse command arguments for explicit base branch** (e.g., "base it on X", "--base X", "base on current branch")
- **If base specified: use it** - explicit arguments override all defaults
- If no base specified: analyze git log and determine appropriate base (usually `develop`)
- If on `develop` with uncommitted changes, ask user about creating feature branch
- Ignore unstaged changes - only include committed/staged changes in PR analysis

### 2. Gather Context (automated analysis)

Collect data:

- Get full diff: `git diff {base}...HEAD`
- Get commit history: `git log {base}..HEAD --oneline`
- Identify affected packages from changed files

Analyze thoroughly (ultrathink about code changes, architectural implications, potential impacts):

- Read through the entire diff carefully
- For context, read any files referenced but not shown in the diff
- Understand purpose and impact of each change
- Identify user-facing changes vs internal implementation details
- Look for breaking changes or migration requirements

### 3. Run Verification

Execute verification commands and track results:

- Run `yarn typecheck` - mark pass/fail
- Run `yarn test` - mark pass/fail
- Run `yarn lint` (if applicable) - mark pass/fail
- Document any failures with explanation

### 4. Generate PR Description

Use this template, filling sections from analysis:

```md
## What problem(s) was I solving?

{problem description}

Related to https://app.clickup.com/t/{taskId} CU-{taskId}

## What user-facing changes did I ship?

{user-facing changes or "Internal changes only"}

## How I implemented it

{technical overview, key decisions, breaking changes}

## How to verify it

### Automated
- [x/blank] `yarn typecheck` - {result}
- [x/blank] `yarn test` - {result}

### Manual Testing
{QA guidance - what to test, edge cases, affected areas}

## Description for the changelog

{concise changelog entry}
```

### 5. User Confirmation

Show draft description and ask:

- ClickUp task ID? (if not detected)
- Reviewers? (use real names from team list)
- Enable auto-merge with squash? (default: yes)
- Any corrections to the generated description?

### 6. Create PR

- Push branch if needed
- Create PR with generated description: `gh pr create --title "..." --body "..."`
- **IMPORTANT**: Use simple quoted strings for `--body`, NOT HEREDOCs (`<<EOF`). Sandbox blocks temp file creation needed for HEREDOCs.
- **ALWAYS** assign PR to current user: `gh pr edit --add-assignee @me`
- Add specified reviewers: `gh pr edit --add-reviewer {username}`
- Enable auto-merge if confirmed: `gh pr merge --auto --squash`

## Team Members

GitHub username to real name mapping:

- **achtan** - David Durika
- **bernard-jigx** - Bernard
- **grant-d** - Grant
- **suzetteJigx** - Suzette Botma
- **xotahal** - Jiri Otahal
- **p-spacek** - Peter Spacek

**Goal**: Auto-generate comprehensive description, run verification, clear QA guidance.
