# Summarize Discussion

Analyze the current conversation and produce a concise prose summary.

## What to Extract

- Problem/need that started the discussion
- Agreed solution/approach
- Key decisions (outcomes only, not deliberation)
- Constraints or requirements
- Open questions (if any)

## Output Style

Write as natural prose - NOT structured with headers/sections. A few paragraphs that flow naturally but contain all essential info.

Example output:

```
We discussed [problem]. The issue is [root cause/context].

Agreed to [solution approach]. Key decisions: [decision 1], [decision 2].

Relevant files: `path/to/file.ts`. Must consider [constraint].

---
Unresolved:
- Q1: [question]? A1: [option], A2: [option], A3: [option]
- Q2: [question]? B1: [option], B2: [option], B3: [option]
```

## Guidelines

- Prose, not bullet lists or headers (except unresolved questions)
- Concise but complete
- Include file paths/technical specifics inline
- Focus on outcomes, skip the back-and-forth
- Keep it scannable - short paragraphs
- End with "Unresolved:" section if there are open questions/ambiguities
- Each unresolved question must have 3+ options: Q1: [question]? A1, A2, A3; Q2: [question]? B1, B2, B3; etc.

## Iterative Loop

When the user answers unresolved questions, re-run this command: incorporate their answers into the summary and print the updated version. Repeat until no unresolved questions remain.
