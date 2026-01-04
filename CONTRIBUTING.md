# Contributing to Document Management Agent

Welcome to the Document Management Agent project! We appreciate your interest in contributing to this project.

## Table of Contents

1. [How to Contribute](#how-to-contribute)
2. [Code of Conduct](#code-of-conduct)
3. [Development Process](#development-process)
4 [Style Guide](#style-guide)
5. [Pull Request Process](#pull-request-process)

## How to Contribute

We welcome contributions of all kinds:
- Bug fixes
- Feature additions
- Documentation improvements
- Code cleanup
- Test case additions

To contribute, please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating in discussions.

## Development Process

1. Ensure you have Python 3.12+ and Node.js 18+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   cd src/frontend
   npm install
   ```
3. Make changes to the codebase
4. Run tests to ensure nothing is broken:
   ```bash
   python -m pytest tests/
   ```

## Style Guide

We follow standard Python and JavaScript conventions. Please ensure your code follows these conventions:
- Python: PEP 8 style guide
- JavaScript: Standard JavaScript conventions
- Comments: Clear and descriptive
- Documentation: Updated as needed

## Pull Request Process

1. All PRs should target the main branch
2. PRs must pass all tests
3. PRs should include documentation updates
4. PRs must be reviewed by team members before merging

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
