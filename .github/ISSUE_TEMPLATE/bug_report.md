---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
What actually happened instead.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Environment
**Desktop/Server:**
 - OS: [e.g., macOS 14.5, Ubuntu 22.04]
 - Python Version: [e.g., 3.11.5]
 - Deployment Method: [e.g., Docker, Local Python]
 - Browser (if web issue): [e.g., Chrome 120, Safari 17]

**Docker Environment (if applicable):**
 - Docker Version: [e.g., 24.0.6]
 - Docker Compose Version: [e.g., 2.20.0]

**Dependencies:**
```
# Output of: pip freeze | grep -E "(streamlit|openai|fastapi|pandas)"
streamlit==1.28.0
openai==2.14.0
...
```

## Logs
<details>
<summary>Error logs (click to expand)</summary>

```
Paste relevant logs here
```
</details>

## Configuration
**`.env` file (redact sensitive info):**
```
AGENT_MODEL=gpt-4
DEBUG=true
# etc.
```

## Additional Context
Add any other context about the problem here.

## Possible Solution
If you have ideas on how to fix this, please share them.

## Related Issues
Link to related issues if any.
