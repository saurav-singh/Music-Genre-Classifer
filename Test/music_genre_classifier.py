import librosa
import numpy as np
import tensorflow.keras as keras

# TODO: Config Variables - Store externally
MODEL_PATH = "../Model/model.h5"
AUDIO_DURATION = 30
SAMPLE_LENGTH = 22050 * AUDIO_DURATION
n_mfcc = 13
n_fft = 2048
hop_length = 512
num_segments = 10
samples_per_segment = int(SAMPLE_LENGTH / num_segments)
num_mfcc_vectors_per_segment = int(np.ceil(samples_per_segment / hop_length))


class _Music_Genre_Classifier:

    GENRES = {
        0: "Blues", 1: "Classical", 2: "Country", 3: "Disco", 4: "Hiphop",
        5: "Jazz", 6: "Metal", 7: "Pop", 8: "Reggae", 9: "Rock"
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
        # if (len(signal) > num_mfcc_vectors_per_segment):
        #     signal = signal[:num_mfcc_vectors_per_segment]
        X = []
        for s in range(num_segments):
            start = samples_per_segment * s
            finish = start + samples_per_segment

            # Extract MFCC
            MFCC = librosa.feature.mfcc(signal[start:finish],
                                        sr=sr,
                                        n_mfcc=n_mfcc,
                                        n_fft=n_fft,
                                        hop_length=hop_length).T

            if len(MFCC) == num_mfcc_vectors_per_segment:
                return MFCC

        # Return MFCC
        return np.array(X)

# TODO: Write better Singleton


def Music_Genre_Classifier():
    if _Music_Genre_Classifier.instance == None:
        _Music_Genre_Classifier.instance = _Music_Genre_Classifier()
        _Music_Genre_Classifier.model = keras.models.load_model(MODEL_PATH)
    return _Music_Genre_Classifier.instance
