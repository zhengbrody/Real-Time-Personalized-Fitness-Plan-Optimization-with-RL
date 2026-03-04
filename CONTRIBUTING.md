# Contributing to ProFit AI

Thank you for your interest in contributing to ProFit AI! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Publishing others' private information
- Trolling or inflammatory comments
- Other conduct which could reasonably be considered inappropriate

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**When filing a bug report, include:**

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior** vs. **actual behavior**
- **Screenshots** (if applicable)
- **Environment details**:
  - OS version
  - Python version
  - Package versions (`pip freeze`)
  - Browser (for web interface issues)

**Example:**

```markdown
**Bug**: AI Coach not responding to queries

**Steps to reproduce**:
1. Open web interface
2. Navigate to AI Coach tab
3. Type "Explain my plan" and click Send
4. No response appears

**Expected**: AI Coach should respond with explanation
**Actual**: Chat shows "AI Coach is offline"

**Environment**:
- macOS 14.5
- Python 3.11.5
- openai==2.14.0
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**When suggesting enhancements, include:**

- **Clear use case**: Why is this enhancement useful?
- **Detailed description**: How should it work?
- **Alternatives considered**: Did you consider other approaches?
- **Additional context**: Screenshots, mockups, or examples

### Contributing Code

We welcome code contributions! Here are the types of contributions we're looking for:

1. **Bug fixes**: Fix reported issues
2. **New features**: Implement features from the roadmap
3. **Performance improvements**: Optimize code for speed/memory
4. **Documentation**: Improve README, docstrings, or guides
5. **Tests**: Add unit tests or integration tests
6. **Refactoring**: Improve code quality without changing functionality

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)

### Setup Steps

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your copy.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR-USERNAME/RL.git
   cd RL
   ```

3. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/ORIGINAL-OWNER/RL.git
   ```

4. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

6. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

7. **Run tests to verify setup**

   ```bash
   pytest
   ```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Use double quotes `"` for strings
- **Imports**: Group in order: standard library, third-party, local
- **Type hints**: Use type hints for function signatures

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black .

# Check formatting without changing files
black --check .
```

### Linting

We use **flake8** for linting:

```bash
# Run linter
flake8 src/ tests/

# Configuration in .flake8 or setup.cfg
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Run type checker
mypy src/
```

### Example Code Style

```python
"""Module docstring explaining what this module does."""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

from src.recommendation.contextual_bandits import ContextualBandit


def get_recommendation(
    user_id: str,
    body_state: Dict[str, float],
    fitness_goal: str = "strength"
) -> Dict[str, any]:
    """
    Generate workout recommendation based on body state.

    Args:
        user_id: Unique identifier for the user
        body_state: Dictionary containing readiness, HRV, sleep, etc.
        fitness_goal: One of "strength", "endurance", "weight_loss"

    Returns:
        Dictionary containing workout_type, intensity, duration_minutes

    Raises:
        ValueError: If body_state is missing required fields
    """
    if not body_state:
        raise ValueError("body_state cannot be empty")

    # Implementation here
    return {
        "workout_type": "Upper Body Strength",
        "intensity": "moderate",
        "duration_minutes": 45
    }
```

## Commit Message Guidelines

We follow **Conventional Commits** format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks (dependencies, build, etc.)
- `perf`: Performance improvements

### Examples

```bash
# Good commit messages
feat(agent): add GPT-4 integration for AI Coach
fix(web): resolve dark mode text contrast issue
docs(readme): add troubleshooting section
test(bandits): add unit tests for Thompson Sampling

# With body
feat(data): add CSV/JSON file upload

Implement file upload feature allowing users to import historical
health data in batch. Supports both CSV and JSON formats with
automatic validation.

Closes #42
```

### Scope

Common scopes:
- `agent`: AI Coach agent
- `web`: Web interface (Streamlit)
- `api`: FastAPI server
- `bandits`: Contextual Bandits / RL
- `data`: Data collection/processing
- `features`: Feature engineering
- `tests`: Test files
- `docs`: Documentation

## Pull Request Process

### Before Submitting

1. **Update from upstream**

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Create feature branch**

   ```bash
   git checkout -b feat/your-feature-name
   ```

3. **Make your changes**

   - Write clean, readable code
   - Add docstrings and comments where needed
   - Follow coding standards

4. **Write/update tests**

   - Add unit tests for new functions
   - Update existing tests if behavior changed
   - Ensure all tests pass: `pytest`

5. **Update documentation**

   - Update README if adding user-facing features
   - Add docstrings to new functions/classes
   - Update CHANGELOG if applicable

6. **Run quality checks**

   ```bash
   # Format code
   black .

   # Run linter
   flake8 src/ tests/

   # Run type checker
   mypy src/

   # Run tests
   pytest
   ```

### Submitting Pull Request

1. **Push to your fork**

   ```bash
   git push origin feat/your-feature-name
   ```

2. **Create Pull Request on GitHub**

   - Use descriptive title following commit message format
   - Fill out the PR template completely
   - Link related issues (e.g., "Closes #123")
   - Add screenshots for UI changes

3. **PR Description Template**

   ```markdown
   ## Description
   Brief description of what this PR does.

   ## Motivation
   Why is this change needed? What problem does it solve?

   ## Changes
   - [ ] Added feature X
   - [ ] Fixed bug Y
   - [ ] Updated documentation

   ## Testing
   How was this tested?

   ## Screenshots (if applicable)
   [Add screenshots here]

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated and passing
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented if unavoidable)
   ```

4. **Respond to review feedback**

   - Address reviewer comments
   - Push additional commits if needed
   - Mark conversations as resolved

5. **Merge approval**

   - At least 1 approval required
   - All CI checks must pass
   - No merge conflicts

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   ├── test_bandits.py
│   ├── test_agent.py
│   └── test_features.py
├── integration/           # Integration tests (slower, multiple components)
│   ├── test_api.py
│   └── test_pipeline.py
└── fixtures/              # Test data and fixtures
    └── sample_data.json
```

### Writing Tests

Use **pytest** framework:

```python
import pytest
from src.recommendation.contextual_bandits import ThompsonSampling


def test_thompson_sampling_initialization():
    """Test Thompson Sampling initializes with correct parameters."""
    ts = ThompsonSampling(n_arms=5, n_features=10)

    assert ts.n_arms == 5
    assert ts.n_features == 10
    assert ts.theta.shape == (5, 10)


def test_thompson_sampling_select_arm():
    """Test arm selection returns valid arm index."""
    ts = ThompsonSampling(n_arms=5, n_features=10)
    context = np.random.randn(10)

    arm = ts.select_arm(context)

    assert 0 <= arm < 5
    assert isinstance(arm, int)


@pytest.mark.parametrize("reward,expected_count", [
    (1.0, 1),
    (0.5, 1),
    (0.0, 1),
])
def test_thompson_sampling_update(reward, expected_count):
    """Test updating model with different reward values."""
    ts = ThompsonSampling(n_arms=5, n_features=10)
    context = np.random.randn(10)
    arm = 2

    ts.update(arm, context, reward)

    assert ts.counts[arm] == expected_count
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_bandits.py

# Run specific test
pytest tests/unit/test_bandits.py::test_thompson_sampling_initialization

# Run with verbose output
pytest -v

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"
```

### Test Coverage

- **Aim for 80%+ coverage** for critical code paths
- **100% coverage** for utility functions
- Focus on **edge cases** and **error handling**

## Documentation

### Code Documentation

- **Docstrings**: All public functions, classes, and modules
- **Format**: Google-style docstrings
- **Type hints**: Include in function signatures

### User Documentation

- **README.md**: High-level overview and quick start
- **QUICK_START.md**: Detailed setup and usage guide
- **API docs**: Document API endpoints and parameters

### Updating Documentation

When adding features:

1. Update relevant markdown files
2. Add examples and screenshots
3. Update troubleshooting section if needed
4. Keep documentation in sync with code

## Questions?

If you have questions:

- **Check existing issues** for similar questions
- **Open a Discussion** on GitHub
- **Ask in Pull Request** if related to specific code

## Recognition

Contributors will be acknowledged in:

- README.md (Contributors section)
- Release notes
- GitHub contributors page

Thank you for contributing to ProFit AI!
