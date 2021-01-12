"""
Linear regression with PyTorch.

Very much inspired by
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/linear_regression/main.py
"""

# https://github.com/pytorch/pytorch/issues/24807
# pylint: disable=not-callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

print(f"PyTorch version: {torch.__version__}")

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU found :)")
else:
    device = torch.device("cpu")
    print("No GPU :(")


def main():
    """Main function"""

    # Hyper-parameters
    input_size = 1
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Toy dataset
    x_train = torch.tensor(
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
        ],
        dtype=torch.float32,
    ).to(device)

    y_train = torch.tensor(
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
        ],
        dtype=torch.float32,
    ).to(device)

    # Linear regression model
    model = nn.Linear(input_size, output_size).to(device)

    # Training configuration
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.5f}")

    # Plot data and predictions
    with torch.no_grad():
        y_pred = model(x_train)
        plt.plot(x_train.cpu(), y_train.cpu(), "ro", label="Data")
        plt.plot(x_train.cpu(), y_pred.cpu(), label="Prediction")
        plt.title("Linear Regression with PyTorch")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
