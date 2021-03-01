import librosa
import numpy as np
import tensorflow.keras as keras

# TODO: Config Variables - Store externally
MODEL_PATH = "model.h5"
AUDIO_DURATION = 30
SAMPLE_LENGTH = 22050 * AUDIO_DURATION
n_mfcc = 13
n_fft = 2048
hop_length = 512


class _Music_Genre_Classifier:

    GENRES = {
        0: "Disco", 1: "Reggae", 2: "Rock", 3: "Pop", 4: "Blues",
        5: "Country", 6: "Jazz", 7: "Classical", 8: "Metal", 9: "Hiphop"
    }
    model = None
    instance = None

    def predict(self, audio_file_path):

        # Preprocess audio file into MFCCs [# segments, # cofficients]
        MFCC = self.preprocess(audio_file_path)

        # Convert MFCC dim 2d -> 4d [# samples, # segments, # cofficients, #channels]
        MFCC = MFCC[np.newaxis, ..., np.newaxis]

        # Predict the genre
        prediction = self.model.predict(MFCC)
        pred_index = np.argmax(prediction)
        pred_genre = self.GENRES[pred_index]

        return pred_genre

    def preprocess(self, audio_file_path):
        # Load audio file
        signal, sr = librosa.load(audio_file_path)

        # Ensure consistency
        if (len(signal) > SAMPLE_LENGTH):
            signal = signal[:SAMPLE_LENGTH]

        # Extract MFCC
        MFCC = librosa.feature.mfcc(signal,
                                    sr=sr,
                                    n_mfcc=n_mfcc,
                                    n_fft=n_fft,
                                    hop_length=hop_length).T

        # Return MFCC
        return MFCC

# TODO: Write better Singleton


def Music_Genre_Classifier():
    if _Music_Genre_Classifier.instance == None:
        _Music_Genre_Classifier.instance = _Music_Genre_Classifier()
        _Music_Genre_Classifier.model = keras.models.load_model(MODEL_PATH)
    return _Music_Genre_Classifier.instance


if __name__ == "__main__":

    magic = Music_Genre_Classifier()

    a = magic.predict("TEST/classic.wav")
    b = magic.predict("TEST/disco.wav")
    c = magic.predict("TEST/pop.wav")

    print(a)
    print(b)
    print(c)
