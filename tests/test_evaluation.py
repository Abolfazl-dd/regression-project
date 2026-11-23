from src.evaluation import evaluate_model


def test_evaluate_model():
    results = evaluate_model([180, 240], [200, 230])

    assert results["mae"] == 15.0
    assert results["mse"] == 250.0