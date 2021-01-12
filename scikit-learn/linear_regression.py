"""
Linear regression with scikit-learn.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model

print(f"scikit-learn version: {sklearn.__version__}")


def train(model, x_train, y_train, num_epochs, learning_rate):
    "Train e scikit-learn model"

    model.fit(x_train, y_train)

    # Plot data and predictions
    y_pred = model.predict(x_train)
    plt.plot(x_train, y_train, "ro", label="Data")
    plt.plot(x_train, y_pred, label="Prediction")
    plt.title(f"Linear Regression with scikit-learn (TODO)")
    plt.legend()
    plt.show()


def main():
    """Main function"""

    # Hyper-parameters
    input_size = 1
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Toy dataset
    x_train = np.array(
        [
            [3.3],
            [4.4],
            [5.5],
            [6.71],
            [6.93],
            [4.168],
            [9.779],
            [6.182],
            [7.59],
            [2.167],
            [7.042],
            [10.791],
            [5.313],
            [7.997],
            [3.1],
        ]
    )

    y_train = np.array(
        [
            [1.7],
            [2.76],
            [2.09],
            [3.19],
            [1.694],
            [1.573],
            [3.366],
            [2.596],
            [2.53],
            [1.221],
            [2.827],
            [3.465],
            [1.65],
            [2.904],
            [1.3],
        ]
    )

    # Linear regression models

    # Using the Normal Equation
    model_normal = linear_model.LinearRegression()

    # Using SGD
    model_sgd = linear_model.SGDRegressor()

    # Train both models
    for model in [model_normal, model_sgd]:
        train(model, x_train, y_train, num_epochs, learning_rate)


if __name__ == "__main__":
    main()
