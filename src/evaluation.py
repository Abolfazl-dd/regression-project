from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return {
        "mae": mae,
        "mse": mse
    }