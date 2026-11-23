from src.data import split_data
from src.model import train_model
from src.evaluation import evaluate_model


def main(X, y):
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    results = evaluate_model(y_test, y_pred)

    print(results)