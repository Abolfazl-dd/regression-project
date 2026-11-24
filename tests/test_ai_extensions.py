# ==============================================================================
# 🔵 [AI ASSISTANT EXTENSIONS & DIAGNOSTIC TESTS — DAY 14]
# این فایل تست‌های تکمیلی تشخیصی روز ۱۴، اعتبارسنجی ابعاد ماتریس و جمع ضرب داخلی را بررسی می‌کند.
# کدهای اصلی کاربر در src/ کاملاً دست‌نخورده باقی مانده‌اند.
# ==============================================================================
import numpy as np
import pytest
from src.data import split_data
from src.model import train_model
from src.evaluation import evaluate_model


def test_math_function_evaluation():
    # f(x) = 2x^2 - 5x + 3 for x = -2 -> 2(-2)^2 - 5(-2) + 3 = 8 + 10 + 3 = 21
    def f(x):
        return 2 * (x ** 2) - 5 * x + 3
    assert f(-2) == 21


def test_math_dot_product_scalar_aggregation():
    # بررسی اصطکاک روز ۱۴: ضرب داخلی حتماً یک اسکالر تجمیعی است، نه آرایه المان به المان
    u = np.array([3, -2, 4])
    v = np.array([2, 5, -1])
    elementwise = u * v  # [6, -10, -4]
    dot_prod = np.dot(u, v)  # 6 - 10 - 4 = -8
    assert not isinstance(dot_prod, np.ndarray) or dot_prod.ndim == 0
    assert dot_prod == -8
    assert np.sum(elementwise) == -8


def test_math_linear_model_and_matrix_dimensions():
    # X = [2, -1], W = [3, 4]^T, b = 5 -> X @ W + b = 6 - 4 + 5 = 7
    X = np.array([[2, -1]])       # (1, 2)
    W = np.array([[3], [4]])      # (2, 1)
    b = 5
    y_hat = X @ W + b             # (1, 1)
    assert y_hat.shape == (1, 1)
    assert y_hat[0, 0] == 7

    # استخراج ابعاد ماتریس وزن مجهول:
    # X = (N, D) = (300, 5), Y_hat = (N, K) = (300, 2) => W must be (D, K) = (5, 2)
    N, D, K = 300, 5, 2
    X_mat = np.ones((N, D))
    W_mat = np.ones((D, K))
    Y_pred = X_mat @ W_mat
    assert Y_pred.shape == (N, K)


def test_ai_pipeline_end_to_end_verification():
    # تست صحت اجرای پایپ‌لاین رگرسیون روی دیتای خطی
    X = np.array([[50], [60], [70], [80], [90]])
    y = np.array([150, 180, 210, 240, 270])

    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    assert "mae" in metrics and "mse" in metrics
    # برای رابطه y = 3x خطا صفر است
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
