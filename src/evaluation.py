


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


def summarize_predictions(records):
    if len(records) == 0:
        raise ValueError("records should not be empty")

    count = len(records)
    total_error = 0
    total_squared_error = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each record must be a dictionary")

        if "actual" not in record or "predicted" not in record:
            raise ValueError("each record must contain actual and predicted")

        actual = record["actual"]
        predicted = record["predicted"]

        if type(actual) not in (int, float) or type(predicted) not in (int, float):
            raise ValueError("actual and predicted must be numbers")

        error = actual - predicted

        total_error += abs(error)
        total_squared_error += error ** 2
    mae = total_error / count
    mse = total_squared_error / count

    return {
        "count": count,
        "mae": mae,
        "mse": mse
    }