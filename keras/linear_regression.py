"""
Linear regression with Keras.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")
print("GPU found :)" if tf.config.list_physical_devices("GPU") else "No GPU :(")


def main():
    """Main function"""

    # Hyper-parameters
    input_size = 1
    output_size = 1
    num_epochs = 60
    learning_rate = 0.001

    # Toy dataset
    x_train = tf.constant(
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

    y_train = tf.constant(
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

    # Using the Sequential API
    model_sequential = keras.Sequential(
        [
            keras.Input(shape=(input_size,)),
            keras.layers.Dense(units=output_size),
        ],
        name="lr_sequential",
    )

    # Using the Functional API
    inputs = keras.Input(shape=(input_size,))
    outputs = keras.layers.Dense(units=output_size)(inputs)
    model_functional = keras.Model(inputs=inputs, outputs=outputs, name="lr_functional")

    # Linear regression models
    model_list = [model_sequential, model_functional]

    # Train both models
    for model in model_list:
        model.summary()

        # Training configuration
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            loss="mean_squared_error",
        )

        # Training loop
        history = model.fit(x_train, y_train, epochs=num_epochs, verbose=0)
        for epoch in range(num_epochs):
            if (epoch + 1) % 5 == 0:
                epoch_loss = history.history["loss"][epoch]
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5f}")

    # Plot data and predictions
    plt.plot(x_train, y_train, "ro", label="Data")
    for model in model_list:
        y_pred = model.predict(x_train)
        plt.plot(x_train, y_pred, label=model.name)
    plt.title("Linear Regression with Keras")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
