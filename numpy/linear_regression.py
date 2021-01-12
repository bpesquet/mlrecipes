"""
Linear regression with NumPy.
"""

import matplotlib.pyplot as plt
import numpy as np

print(f"NumPy version: {np.__version__}")


def mean_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error between two vectors"""
    return np.square(y_true - y_pred).mean()


def main():
    """Main function"""

    # Hyper-parameters
    input_size = 1
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

    num_samples = x_train.shape[0]

    # Add x0 = 1 to each sample
    X = np.c_[np.ones((num_samples, 1)), x_train]
    assert X.shape == (num_samples, input_size + 1)

    # Init weights (including bias term)
    weights_gd = np.random.randn(input_size + 1, 1)
    assert weights_gd.shape == (input_size + 1, 1)

    # Training loop
    for epoch in range(num_epochs):
        # Gradient computation
        gradients = 2 / num_samples * X.T @ (X @ weights_gd - y_train)

        # Weights update
        weights_gd = weights_gd - learning_rate * gradients

        if (epoch + 1) % 5 == 0:
            epoch_loss = mean_squared_error(y_train, X @ weights_gd)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5f}")

    # Plot data and predictions
    y_pred = X @ weights_gd
    plt.plot(x_train, y_train, "ro", label="Data")
    plt.plot(x_train, y_pred[:, 1], label="Batch GD")
    plt.title("Linear Regression with NumPy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
