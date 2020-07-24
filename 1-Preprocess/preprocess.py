import os
import math
import json
import librosa


DATASET_PATH = "../dataset"
JSON_PATH = "../data.json"
SAMPLE_RATE = 22050
DURATION = 30  # Duration of audio (seconds)
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def saveMFCC(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    # Data Storage
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        # Ignore root path
        if dirpath == dataset_path:
            continue

        # Store Genre
        genre = dirpath.split("\\")[-1]
        data["mapping"].append(genre)
        print("\nProcessing: {}".format(genre))

        # Process audio files for specific genre
        for f in filenames:
            # Load audio file
            audio_file = os.path.join(dirpath, f)
            signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE)

            # Process segments extracting mfcc then store data
            for s in range(num_segments):
                start = samples_per_segment * s
                finish = start + samples_per_segment

                mfcc = librosa.feature.mfcc(
                    signal[start:finish],
                    sr=sr,
                    n_mfcc=n_mfcc,
                    n_fft=n_fft,
                    hop_length=hop_length).T

                if len(mfcc) == num_mfcc_vectors_per_segment:
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    # print("{}, segment:{}".format(audio_file, s+1))

    with open(json_path, "w+") as F:
        json.dump(data, F, indent=4)


saveMFCC(DATASET_PATH, JSON_PATH, num_segments=10)
