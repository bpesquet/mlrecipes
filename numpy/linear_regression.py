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

    num_samples = x_train.shape[0]

    # Add x0 = 1 to each sample for bias weight
    x_train_const = np.c_[np.ones((num_samples, 1)), x_train]
    assert x_train_const.shape == (num_samples, input_size + 1)

    # Set weights (including bias term)
    weights_normal = (
        np.linalg.inv(x_train_const.T @ x_train_const) @ x_train_const.T @ y_train
    )
    assert weights_normal.shape == (input_size + 1, 1)

    # Init weights (including bias term)
    weights_sgd = np.random.randn(input_size + 1, 1)
    assert weights_sgd.shape == (input_size + 1, 1)

    model_list = {"Normal Equation": weights_normal, "SGD": weights_sgd}

    # Training loop
    for epoch in range(num_epochs):
        # Randomly select one data sample
        sample_index = np.random.randint(num_samples)
        x_sample = x_train_const[[sample_index]]
        assert x_sample.shape == (1, input_size + 1)
        y_sample = y_train[[sample_index]]
        assert y_sample.shape == (1, 1)

        # Gradient computation
        gradients = 2 * x_sample.T @ (x_sample @ weights_sgd - y_sample)
        assert gradients.shape == (input_size + 1, 1)

        # Weights update
        weights_sgd -= learning_rate * gradients

        if (epoch + 1) % 5 == 0:
            epoch_loss = mean_squared_error(y_train, x_train_const @ weights_sgd)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5f}")

    # Plot data and predictions
    plt.plot(x_train, y_train, "ro", label="Data")
    for name, weights in model_list.items():
        y_pred = x_train_const @ weights
        final_loss = mean_squared_error(y_train, y_pred)
        print(f"Final loss for {name}: {final_loss:.5f}")
        plt.plot(x_train, y_pred, label=name)
    plt.title("Linear Regression with NumPy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
