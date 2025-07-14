# Contributing to DocuScope Corpus Analysis & Concordancer

Thank you for your interest in contributing to DocuScope CA! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Code Style](#code-style)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your feature or bugfix

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Git

### Installation

1. Clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/docuscope-ca-online.git
cd docuscope-ca-online
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up development configuration:
   - Copy `webapp/config/options.toml` to create your local configuration
   - Set `desktop_mode = true` for local development

5. Run the application:
```bash
streamlit run webapp/index.py
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-analysis-tool`
- `fix/memory-leak-in-processing`
- `docs/update-installation-guide`

### Commit Messages

Write clear, descriptive commit messages:
- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Example:
```
Add support for custom DocuScope dictionaries

- Implement dictionary loading from user files
- Add validation for dictionary format
- Update UI to include dictionary selection
- Add tests for new functionality

Fixes #123
```

## Testing

### Running Tests

We use pytest for testing. Run the test suite with:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=webapp tests/
```

### Writing Tests

- Write tests for all new features
- Include both unit and integration tests
- Test edge cases and error conditions
- Maintain test coverage above 80%

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Tests with external dependencies
├── performance/    # Performance benchmarks
└── streamlit/      # UI workflow tests
```

## Pull Request Process

1. **Update Documentation**: Ensure any new features are documented
2. **Add Tests**: Include appropriate tests for your changes
3. **Check Code Style**: Run linting and formatting tools
4. **Update CHANGELOG**: Add an entry describing your changes
5. **Create Pull Request**: Use the PR template and provide clear description

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
- [ ] PR description clearly explains the changes

### Review Process

- All PRs require at least one review
- Maintainers will provide feedback within 48-72 hours
- Address review comments promptly
- Be open to suggestions and constructive criticism

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the issue
- **Steps to Reproduce**: Detailed steps to reproduce the bug
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: OS, Python version, browser (if applicable)
- **Screenshots**: If applicable, add screenshots
- **Log Output**: Relevant error messages or logs

### Feature Requests

For feature requests, please include:

- **Problem Statement**: What problem does this solve?
- **Proposed Solution**: How should this feature work?
- **Alternatives**: What alternatives have you considered?
- **Use Cases**: Who would benefit from this feature?

### Issue Labels

We use labels to categorize issues:
- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Maximum line length: 88 characters (Black formatter)

### Formatting Tools

We use these tools to maintain code quality:

```bash
# Format code
black webapp/ tests/

# Sort imports
isort webapp/ tests/

# Lint code
flake8 webapp/ tests/

# Type checking
mypy webapp/
```

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Update docstrings when changing function signatures
- Use Google-style docstrings

Example:
```python
def process_corpus(texts: List[str], model_name: str) -> Dict[str, Any]:
    """Process a corpus with the specified model.
    
    Args:
        texts: List of text documents to process
        model_name: Name of the spaCy model to use
        
    Returns:
        Dictionary containing processing results with keys:
        - 'tokens': Token-level annotations
        - 'docs': Document-level statistics
        - 'errors': Any processing errors encountered
        
    Raises:
        ValueError: If model_name is not available
        ProcessingError: If corpus processing fails
    """
```

## Development Guidelines

### Architecture Principles

- **Modularity**: Keep components loosely coupled
- **Testability**: Write code that's easy to test
- **Performance**: Consider memory usage for large corpora
- **User Experience**: Maintain responsive UI
- **Configuration**: Use configuration files for settings

### Adding New Features

1. **Design First**: Discuss major changes in issues before implementing
2. **Start Small**: Break large features into smaller, reviewable chunks
3. **Document APIs**: Update documentation for any new interfaces
4. **Consider Backwards Compatibility**: Avoid breaking existing functionality

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## Getting Help

- **Documentation**: Check the [official documentation](https://browndw.github.io/docuscope-docs/)
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md for significant contributions
- README.md for major features
- Academic citations where appropriate

Thank you for contributing to DocuScope CA! Your contributions help make textual analysis more accessible to researchers and students worldwide.
