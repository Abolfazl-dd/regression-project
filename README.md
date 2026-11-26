# Regression Project

A small Python machine-learning project that trains a Linear Regression model, generates predictions, and evaluates its performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE).

This project also includes input validation, prediction-summary functionality, and automated tests using pytest.

## Features

- Split datasets into training and testing sets.
- Validate feature dimensions before splitting data.
- Train a Linear Regression model using scikit-learn.
- Generate predictions on unseen test data.
- Evaluate model performance using MAE and MSE.
- Calculate prediction summaries from actual and predicted values.
- Validate prediction records and reject invalid inputs.
- Test individual functions and the complete regression pipeline.

## Project Structure

```text
regression-project/
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── evaluation.py
│   └── main.py
│
├── tests/
│   ├── test_data.py
│   ├── test_evaluation.py
│   └── test_ai_extensions.py
│
├── .vscode/
│   └── settings.json
│
├── debugging_log.md
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

- Python
- NumPy
- scikit-learn
- pytest

## Installation

Clone the repository:

```bash
git clone https://github.com/Abolfazl-dd/regression-project.git
```

Navigate to the project directory:

```bash
cd regression-project
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Run the main application from the project root:

```bash
python -m src.main
```

The program uses a small example dataset, splits it into training and testing sets, trains a Linear Regression model, generates predictions, and prints the evaluation results.

## Evaluation Metrics

The project uses two common regression metrics:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values.

**Mean Squared Error (MSE)**

Measures the average squared difference between actual and predicted values. Larger errors have a greater effect on this metric.

Both metrics are implemented in `src/evaluation.py`.

## Prediction Summary

The `summarize_predictions()` function accepts a list of dictionaries containing actual and predicted values.

Example:

```python
records = [
    {"actual": 100, "predicted": 90},
    {"actual": 80, "predicted": 85},
    {"actual": 60, "predicted": 65}
]

result = summarize_predictions(records)

print(result)
```

Expected output:

```python
{
    "count": 3,
    "mae": 6.666666666666667,
    "mse": 50.0
}
```

The function also validates its input and raises `ValueError` when records are empty, have an invalid structure, or contain non-numeric values.

## Running Tests

Run the complete test suite from the project root:

```bash
python -m pytest -v
```

The tests cover:

- Dataset splitting and input validation.
- Regression model evaluation.
- Prediction summary calculations.
- Invalid input handling.
- Mathematical operations and matrix dimensions.
- End-to-end regression pipeline verification.

## Learning Objectives

This project is part of my ongoing study of:

- Python programming and modular code organization.
- NumPy and numerical operations.
- Data preprocessing and validation.
- Linear Regression and machine learning.
- Model evaluation and error analysis.
- Unit testing and debugging.
- Mathematical foundations of machine learning.

## Author

**Abolfazl**

GitHub: [Abolfazl-dd](https://github.com/Abolfazl-dd)
