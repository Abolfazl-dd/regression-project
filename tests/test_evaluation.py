
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "mae": mae,
        "mse": mse
        }

def test_evaluate_model():
    
    evaluate = evaluate_model([180, 240],[200, 230]) 
    assert evaluate["mae"] == 15.0
    assert evaluate["mse"] == 250.0
