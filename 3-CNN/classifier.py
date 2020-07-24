import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


def load_data(dataset_path):
    with open(dataset_path, "r") as F:
        data = json.load(F)
    X = np.array(data["mfcc"])
    Y = np.array(data["labels"])
    return X, Y


def prepare_data(test_size, validation_size):
    # Load data
    data = "../data.json"
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

    # 1st convolutional layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd convolutional layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd convolutional layer
    model.add(keras.layers.Conv2D(
        32, (2, 2), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # Flatten the output then feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # Output layer (10 genres)
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def predict(model, X, Y):
    genre = {
        0: "Disco", 1: "Reggae", 2: "Rock", 3: "Pop", 4: "Blues",
        5: "Country", 6: "Jazz", 7: "Classical", 8: "Metal", 9: "Hiphop"}

    X = X[np.newaxis, ...]
    prediction = model.predict(X)

    index = np.argmax(prediction, axis=1)
    predicted_genre = genre[index]
    expected_genre = genre[Y]

    print("Predicted Genre {}".format(predicted_genre))
    print("Expected Genre {}".fomat(expected_genre))


if __name__ == "__main__":
    # Create train , test set
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_data(
        0.25, 0.2)

    # Build CNN Model
    input_shape = tuple(X_train.shape[1:])
    model = build_model(input_shape)

    # Optimize the model
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Train CNN Model
    model.fit(X_train, Y_train, validation_data=(
        X_validation, Y_validation), batch_size=32, epochs=30)

    # Evaluate CNN Model
    test_error, test_accuracy = model.evaluate(X_test, Y_test, verbose=1)
    print("Accuracy = {}".format(test_accuracy))

    # Make Prediction
    X = X_test[100]
    Y = Y_test[100]
    predict(model, X, Y)
