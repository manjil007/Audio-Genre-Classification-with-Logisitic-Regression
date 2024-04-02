import os
import numpy as np
import librosa
import pandas as pd


def extract_features(file_path, n_mfcc=40, n_segments=5):
    """
    Extracts features from an audio file.

    Parameters:
    - file_path: Path to the audio file.
    - n_mfcc: Number of MFCCs to extract.
    - n_segments: Number of segments to divide the audio into.

    Returns:
    - features: Extracted features.
    """
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    audio_duration = librosa.get_duration(y=audio, sr=sample_rate)

    segment_len = int(audio_duration / n_segments)
    hop_length = int(segment_len * sample_rate)

    mfccs = []
    chroma_stft = []

    features = []

    for i in range(n_segments):
        start = i * segment_len
        end = (i + 1) * segment_len

        segment = audio[start:end]

        mfccs = (np.mean(librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, hop_length=hop_length).T, axis=0))
        chroma_stft = (np.mean(librosa.feature.chroma_stft(y=segment, sr=sample_rate).T, axis=0))

        features.extend(np.hstack((mfccs, chroma_stft)))

    return features


def process_dataset_for_training(root_dir_train='data/train', n_mfcc=40, n_segments=5):
    features = []
    labels = []
    genres = os.listdir(root_dir_train)
    for genre in genres:
        genre_path = os.path.join(root_dir_train, genre)
        for filename in os.listdir(genre_path):
            if filename.endswith('.au'):
                file_path = os.path.join(genre_path, filename)
                try:
                    features.append(extract_features(file_path, n_mfcc=n_mfcc, n_segments=n_segments))
                    labels.append(genre)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    features = np.array(features)
    labels = np.array(labels)

    features_shape = features.shape
    labels_shape = labels.shape

    labels = labels.reshape(-1, 1)

    combined_data = np.hstack((features, labels))

    feature_columns = ['feature_' + str(i + 1) for i in range(len(features[1]))]
    df = pd.DataFrame(combined_data, columns=feature_columns + ['Genre'])

    df.to_csv('extracted_dataset.csv', index=False)


process_dataset_for_training(root_dir_train='data/train', n_mfcc=40, n_segments=5)
