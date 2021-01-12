"""
Linear regression with scikit-learn.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model, metrics

print(f"scikit-learn version: {sklearn.__version__}")


def main():
    """Main function"""

    # Hyper-parameters
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
            1.7,
            2.76,
            2.09,
            3.19,
            1.694,
            1.573,
            3.366,
            2.596,
            2.53,
            1.221,
            2.827,
            3.465,
            1.65,
            2.904,
            1.3,
        ]
    )

    # Using the Normal Equation
    model_normal = linear_model.LinearRegression()

    # Using SGD
    model_sgd = linear_model.SGDRegressor(
        max_iter=num_epochs, learning_rate="constant", eta0=learning_rate
    )

    # Linear regression models
    model_list = [model_normal, model_sgd]

    # Train both models
    for model in model_list:
        model.fit(x_train, y_train)
        final_loss = metrics.mean_squared_error(y_train, model.predict(x_train))
        print(f"Final loss for {type(model).__name__}: {final_loss:.5f}")

    # Plot data and predictions
    plt.plot(x_train, y_train, "ro", label="Data")
    for model in model_list:
        y_pred = model.predict(x_train)
        plt.plot(x_train, y_pred, label=type(model).__name__)
    plt.title("Linear Regression with scikit-learn")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
