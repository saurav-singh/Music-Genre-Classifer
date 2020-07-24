import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split


def load_data(dataset_path):
    with open(dataset_path, "r") as F:
        data = json.load(F)
    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])

    return X, Y


def plot_result(result):
    _, axis = plt.subplots(2)

    # Accuracy Subplot
    axis[0].plot(result.history["accuracy"], label="train accuracy")
    axis[0].plot(result.history["val_accuracy"], label="test accuracy")
    axis[0].set_ylabel("Accuracy")
    axis[0].legend(loc="lower right")
    axis[0].set_title("Accuracy eval")

    # Loss Subplot
    axis[1].plot(result.history["loss"], label="train loss")
    axis[1].plot(result.history["val_loss"], label="test loss")
    axis[1].set_xlabel("Epoch")
    axis[1].set_ylabel("Loss")
    axis[1].legend(loc="upper right")
    axis[1].set_title("Loss eval")

    # Display Plots
    plt.show()


if __name__ == "__main__":

    # Load Data
    data = "../data.json"
    inputs, targets = load_data(data)

    # Split train and test set
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.3)

    # Build Network architecture
    model = keras.Sequential([
        # Input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st Hidden layer
        keras.layers.Dense(512, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd Hiddel layer
        keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd Hidden Layer
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # Output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # Compile Network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    # Train Network
    result = model.fit(inputs_train, targets_train,
                       validation_data=(inputs_test, targets_test),
                       batch_size=32, epochs=100,)

    # Plot Result
    plot_result(result)
