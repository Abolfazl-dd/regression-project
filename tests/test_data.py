import numpy as np
import pytest

from src.data import split_data


def test_split_data():
    X = np.array([[50], [60], [70], [80], [90]])
    y = np.array([150, 180, 210, 240, 270])

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) == 4
    assert len(X_test) == 1

    assert len(y_train) == 4
    assert len(y_test) == 1

    assert X_train.shape[1] == 1
    assert X_test.shape[1] == 1





def test_split_data_rejects_mismatched_lengths():
    X = np.array([[50], [60], [70]])
    y = np.array([150, 180])

    with pytest.raises(ValueError):
        split_data(X, y)




def test_split_data_rejects_1d_features():
    X = np.array([50, 60, 70, 80, 90])
    y = np.array([150, 180, 210, 240, 270])

    with pytest.raises(ValueError):
        split_data(X, y)