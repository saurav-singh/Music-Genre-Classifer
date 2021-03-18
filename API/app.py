import os
import sys
import uuid
from flask import Flask
from flask import request
from music_genre_classifier import Music_Genre_Classifier

app = Flask(__name__)


CLASSIFIER = None
ROOT_FILE = "Audio_Files"


@app.route("/predict", methods=["POST"])
def predict():
    # Load and save audio file
    audio_file = request.files["file"]
    unique_file_name = str(uuid.uuid1()) + ".wav"
    file_name = os.path.join(ROOT_FILE, unique_file_name)
    audio_file.save(file_name)

    # Make Prediction
    prediction = CLASSIFIER.predict(file_name)

    # Remove audio file
    os.remove(file_name)

    # Return Prediction
    return {"genre": prediction}


if __name__ == "__main__":
    # Initialize Classifer
    CLASSIFIER = Music_Genre_Classifier()
    app.run(debug=False)
