# Before commit

Please analyze the uncommitted changes in this repository and provide a comprehensive code review. DO NOT make any changes to the code - only provide suggestions for the human to review and approve.

## IMPORTANT: Analysis Approach

1. First, identify all files that have uncommitted changes
2. For each modified file, analyze the ENTIRE CURRENT VERSION of the file, not just the diff
3. Review the complete context of the code, including:
   - How new/modified code interacts with existing code
   - Whether existing patterns in the file are being followed
   - Overall file structure and organization
   - Dependencies and imports used throughout the file

## Review Scope

For each file with uncommitted changes, examine the COMPLETE FILE for:

### 1. Code Quality & Best Practices

- Code readability and clarity
- Naming conventions (variables, functions, classes, files)
- Code duplication and DRY principle violations
- Function/method complexity and length
- Proper error handling and edge cases
- Resource management (memory leaks, file handles, connections)
- Dead code or unused imports/variables
- Magic numbers and hardcoded values
- Code formatting consistency
- Consistency with existing code patterns in the file

### 2. Design Patterns & Architecture

- SOLID principles adherence
- Appropriate design pattern usage
- Separation of concerns
- Coupling and cohesion
- Abstraction levels
- Dependency injection opportunities
- Interface segregation
- How the changes fit into the overall architecture

### 3. Security

- Input validation and sanitization
- SQL injection vulnerabilities
- XSS vulnerabilities
- Authentication/authorization issues
- Sensitive data exposure (API keys, passwords)
- Cryptographic weaknesses
- Path traversal risks
- CORS and security headers

### 4. Performance

- Algorithm efficiency (time/space complexity)
- Database query optimization
- Caching opportunities
- Unnecessary loops or iterations
- Memory usage patterns
- Blocking operations that should be async
- Performance impact on the entire module/file

### 5. Testing

- Test coverage gaps for the entire file/module
- Missing edge case tests
- Test quality and assertions
- Mock/stub usage appropriateness
- Test naming and organization
- Whether new changes are properly tested

### 6. Documentation

- Missing or outdated comments
- Function/class documentation
- README updates needed
- Inline comment quality
- Whether documentation matches the current implementation

### 7. Dependencies

- Outdated dependencies
- Security vulnerabilities in dependencies
- Unnecessary dependencies
- License compatibility issues
- Proper import organization

### 8. Version Control

- File organization
- Binary files that shouldn't be committed
- Sensitive files (.env, secrets)

## Output Format

Categorize all findings by severity:

### 🔴 CRITICAL (Must fix before merge)

- Security vulnerabilities
- Data loss risks
- Breaking changes
- Major bugs
- Issues that affect system stability

### 🟡 WARNING (Should fix)

- Performance issues
- Best practice violations
- Minor bugs
- Technical debt
- Code inconsistencies

### 🔵 INFO (Consider improving)

- Style improvements
- Documentation enhancements
- Refactoring opportunities
- Minor optimizations
- Code modernization suggestions

For each finding, provide:
**File and line number(s)**: [exact file path and specific line numbers]
**Context**: [what the code is doing in this section]
**Issue description**: [clear explanation of what's wrong]
**Why it's a problem**: [specific impact and risks]
**Suggested fix**:

```[language]
// Current code
[show problematic code]

// Suggested improvement
[show corrected code]
```

**References**: [links to documentation, best practices, or examples if applicable]

## Analysis Instructions

1. Start by listing all files with uncommitted changes
2. For each file, load and analyze the COMPLETE CURRENT VERSION
3. Don't just look at what changed - look at how the changes affect the entire file
4. Consider the broader context of each file and its role in the system
5. Check if new code follows existing patterns in the codebase
6. Verify that changes don't break existing functionality

## Summary

After analyzing all files, provide:

- List of all files reviewed
- Total findings by severity (Critical: X, Warning: Y, Info: Z)
- Overall code quality assessment
- Top 3 priority fixes that must be addressed
- Estimated effort for addressing all issues (hours/days)
- Positive aspects worth noting (what was done well)
