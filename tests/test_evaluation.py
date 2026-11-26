from src.evaluation import evaluate_model
import pytest
from src.evaluation import summarize_predictions


def test_evaluate_model():
    results = evaluate_model([180, 240], [200, 230])

    assert results["mae"] == 15.0
    assert results["mse"] == 250.0




def test_summarize_predictions():
    records = [
        {"actual": 100, "predicted": 90},
        {"actual": 80, "predicted": 85},
        {"actual": 60, "predicted": 65}
    ]

    result = summarize_predictions(records)

    assert result["count"] == 3
    assert result["mae"] == pytest.approx(20 / 3)
    assert result["mse"] == pytest.approx(50)




def test_summarize_predictions_rejects_empty_records():
    with pytest.raises(ValueError):
        summarize_predictions([])


def test_summarize_predictions_rejects_missing_predicted():
    records = [
        {"actual": 100, "predicted": 90},
        {"actual": 80, "predicted": 85},
        {"actual": 60}]

    with pytest.raises(ValueError):
        summarize_predictions(records)


def test_summarize_predictions_rejects_non_dictionary_record():
    records = [
        {"actual": 100, "predicted": 90},
        {"actual": 80, "predicted": 85},
        [60,2]
        ]

    with pytest.raises(ValueError):
        summarize_predictions(records)


def test_summarize_predictions_rejects_non_numeric_prediction():
    records = [
        {"actual": 100, "predicted": 90},
        {"actual": 80, "predicted": "85"},
        ]

    with pytest.raises(ValueError):
        summarize_predictions(records)



def test_summarize_predictions_rejects_non_numeric_actual():
    records = [
        {"actual": 100, "predicted": 90},
        {"actual": "80", "predicted": 85},
        ]

    with pytest.raises(ValueError):
        summarize_predictions(records)