# AutoyML

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Build Status](https://travis-ci.com/prouhard/autoyml-homework.svg?token=kyCztgpxTDJaAsiPLynz&branch=master)](https://travis-ci.com/prouhard/autoyml-homework)

**AutoyML** is a toy automatic machine learning library made for learning purposes.

### Notes

The aim was to develop a Keras model which could adapt its input layer based on the data, and automatically tune its core architecture to best fit it.

The actual class with the required methods, `NeuralNetworkModel`, can be found [here](autoyml/neural_network_model.py).

It leverages the [Keras Tuner](https://keras-team.github.io/keras-tuner/) library to tune the hyper parameters.

The unit tests can be found [here](tests/test_autoyml.py) in the tests folder, and are scheduled to run with Travis CI.

### Installation

You will need [`pipenv`](https://github.com/pypa/pipenv) to easily install all the required dependencies.

```bash
python3 -m pip install pipenv
PIPENV_VENV_IN_PROJECT=true pipenv install --dev
```

### Useful commands

| `pipenv run <command>` | Description                |
| ---------------------- | -------------------------- |
| `pytest tests`         | Run the test suite.        |
| `flake8 autoyml`       | Run the flake8 linter.     |
| `mypy autoyml`         | Run the MyPy type checker. |

### Application structure

```
.
├── autoyml                      # Application source code
│   ├── abstract_model.py          # Definition of the model interface
│   ├── decorators.py              # Decorator checking that the model can predict
│   ├── errors.py                  # Custom errors
│   ├── hypermodel.py              # Tunable neural network model and custom Keras Tuner's `KerasHyperModel`
│   ├── neural_network_model.py    # *** Actual wrapper class, implementing the required methods ***
│   ├── preprocessing.py           # Collection of helper functions to manipulate the data
│   └── tuner.py                   # Custom `Tuner` class with cross-validation evaluation
└── tests                          # Application tests
    └── test_autoyml.py          # Unit tests

```
