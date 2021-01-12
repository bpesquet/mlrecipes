[![Build status](https://github.com/bpesquet/mlrecipes/workflows/build/badge.svg)](https://github.com/bpesquet/mlrecipes/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Machine Learning recipes

Minimalist, self-contained implementations of various Machine Learning algorithms, using the simplest possible dataset for each task.

## Table of contents

### Linear Regression

- [PyTorch](pytorch/linear_regression.py) | [Keras](keras/linear_regression.py) | [scikit-learn](scikit-learn/linear_regression.py) | [NumPy](numpy/linear_regression.py)

## Development notes

### Checking the code

This project uses the following tools:

- [black](https://github.com/psf/black) for code formatting.
- [pylint](https://www.pylint.org/) for linting.

Run the following command in project root folder to check the codebase.

```bash
> pylint -d duplicate-code **/*.py  # linting
```
