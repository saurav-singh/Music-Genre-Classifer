from music_genre_classifier import Music_Genre_Classifier

if __name__ == "__main__":

    classifier = Music_Genre_Classifier()

    a = classifier.predict("Music/classical.wav")
    b = classifier.predict("Music/hiphop.wav")
    c = classifier.predict("Music/pop.wav")

    print(a, b, c)
