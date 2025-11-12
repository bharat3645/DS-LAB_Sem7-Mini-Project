# Contributing to Crime Analytics Dashboard

First off, thank you for considering contributing to the Crime Analytics Dashboard! It's people like you that make this project better for everyone.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps which reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps**
* **Explain which behavior you expected to see instead and why**
* **Include screenshots and animated GIFs if possible**
* **Include your environment details** (OS, Python version, browser, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the steps**
* **Describe the current behavior** and **explain the expected behavior**
* **Explain why this enhancement would be useful**

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the Python style guide (PEP 8)
* Include thoughtful commit messages
* Update documentation when needed
* Add tests when adding new features
* End all files with a newline

## Development Process

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Click the 'Fork' button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/crime-analytics-project.git
   cd crime-analytics-project
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Code Style

We follow PEP 8 style guide for Python code. Key points:

* Use 4 spaces for indentation (not tabs)
* Limit lines to 79 characters for code, 72 for comments
* Use meaningful variable names
* Add docstrings to functions and classes
* Use type hints where appropriate

**Example:**
```python
def calculate_crime_rate(total_crimes: int, population: int) -> float:
    """
    Calculate crime rate per 1000 population.
    
    Args:
        total_crimes: Total number of crimes
        population: Total population
        
    Returns:
        Crime rate per 1000 people
    """
    return (total_crimes / population) * 1000
```

### Testing

Before submitting a PR, make sure:

1. **Run the app locally**
   ```bash
   streamlit run streamlit_app2.py
   ```

2. **Test all features you modified**
   - Navigate through all pages
   - Test filters and interactions
   - Verify visualizations render correctly

3. **Check for errors**
   - No Python exceptions in console
   - No browser console errors
   - Data loads correctly

### Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

**Good examples:**
```
Add crime trend visualization to dashboard
Fix data loading issue for missing persons
Update README with deployment instructions
Refactor clustering analysis for better performance
```

**Bad examples:**
```
fixed stuff
updates
minor changes
```

### Project Structure Guidelines

When adding new features:

1. **For data processing**: Add to `data_preprocessing.py`
2. **For ML models**: Add to `ml_models.py`
3. **For DL models**: Add to `deep_learning.py`
4. **For visualizations**: Add to relevant dashboard file
5. **For utilities**: Create a new `utils.py` file

### Documentation

* Update README.md if you change functionality
* Add inline comments for complex logic
* Update docstrings when modifying functions
* Create/update examples in documentation

### What to Contribute

**Good First Issues:**
* Documentation improvements
* Adding more visualizations
* Improving error messages
* Adding data validation
* Writing tests

**Advanced Contributions:**
* New ML models or techniques
* Performance optimizations
* New analysis features
* Geographic map integration
* Real-time data updates

### Areas We Need Help

1. **Testing**
   - Unit tests for data preprocessing
   - Integration tests for ML pipeline
   - UI/UX testing

2. **Documentation**
   - Tutorial videos
   - More code examples
   - API documentation

3. **Features**
   - Time series forecasting
   - Interactive maps
   - Export functionality
   - Multi-language support

4. **Performance**
   - Optimize data loading
   - Improve model training speed
   - Reduce memory usage

## Review Process

1. **Submit PR** with clear description
2. **Automated checks** run (if configured)
3. **Maintainer review** within 3-5 days
4. **Address feedback** if requested
5. **Merge** once approved

## Recognition

Contributors will be:
* Listed in README.md acknowledgments
* Mentioned in release notes
* Credited in relevant documentation

## Questions?

Feel free to:
* Open an issue with your question
* Start a discussion on GitHub
* Contact the maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to making crime analytics more accessible and insightful! 🎉
