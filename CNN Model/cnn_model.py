import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense,  Dropout

from tensorflow.keras.optimizers import Adam

MODEL_OUTPUT = "model.h5"


def load_data(dataset_path):
    with open(dataset_path, "r") as F:
        data = json.load(F)
    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])
    return X, Y


def prepare_data(test_size, validation_size):
    # Load data
    data = "Data.json"
    X, Y = load_data(data)

    # Split into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size)

    # Split into train and validation set
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=validation_size)

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


def build_model(input_shape):
    # Create Model
    model = keras.Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape,
                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    # Convolutional Layer 2
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape,
                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    # Convolutional Layer 3
    model.add(Conv2D(32, (2, 2), activation="relu", input_shape=input_shape,
                     kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())

    # Flatten into dense layer
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))

    # Output Layer
    NUM_GENRES = 10
    model.add(Dense(NUM_GENRES, activation="softmax"))

    # Compile the Model
    learning_rate = 0.001
    optimizer = Adam(learning_rate=learning_rate)
    loss = "sparse_categorical_crossentropy"

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    model.summary()

    return model


if __name__ == "__main__":
    # Create train , test set
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_data(
        0.25, 0.2)

    # Build CNN Model
    input_shape = tuple(X_train.shape[1:])
    model = build_model(input_shape)

    # Train CNN Model
    model.fit(X_train, Y_train, validation_data=(
        X_validation, Y_validation), batch_size=32, epochs=30)

    # Evaluate CNN Model
    test_error, test_accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Error : {test_error} | Test Accuracy {test_accuracy}")

    # Save the Model
    model.save(MODEL_OUTPUT)
