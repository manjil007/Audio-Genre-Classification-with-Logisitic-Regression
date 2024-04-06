import os
import numpy as np
import librosa
import pandas as pd


def extract_features(file_path, n_mfcc=20):
    """
    Extracts features from an audio file using multiple sampling rates and averages the features.

    Parameters:
    - file_path: Path to the audio file.
    - n_mfcc: Number of MFCCs to extract.

    Returns:
    - avg_features: Averaged extracted features across different sampling rates.
    """
    sampling_rates = np.arange(15000, 50001, 5000)  # Array of sampling rates from 5000 to 50000
    feature_list = []  # List to store feature arrays for each sampling rate

    for sr in sampling_rates:
        audio, sample_rate = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
        stft = np.abs(librosa.stft(audio))
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).T, axis=0)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        rmse = np.mean(librosa.feature.rms(y=audio).T, axis=0)
        # Append the computed features for the current sampling rate to the list
        features = np.hstack((mfccs, chroma_stft, spectral_contrast, spectral_bandwidth, zero_crossing_rate, rmse))
        feature_list.append(features)

    # Convert the list of features to a NumPy array and average across all sampling rates
    avg_features = np.mean(np.array(feature_list), axis=0)

    return avg_features


def process_dataset_for_training(root_dir_train):
    features = []
    labels = []
    genres = os.listdir(root_dir_train)
    for genre in genres:
        genre_path = os.path.join(root_dir_train, genre)
        for filename in os.listdir(genre_path):
            if filename.endswith('.au'):
                file_path = os.path.join(genre_path, filename)
                try:
                    segment_features = (extract_features(file_path))
                    features.append(segment_features)
                    labels.append(genre)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    features = np.array(features)
    labels = np.array(labels)

    labels = labels.reshape(-1, 1)

    combined_data = np.hstack((features, labels))

    feature_columns = ['feature_' + str(i + 1) for i in range(len(features[1]))]
    df = pd.DataFrame(combined_data, columns=feature_columns + ['Genre'])

    name = 'train_features.csv'

    df.to_csv(name, index=False)


def process_dataset_for_testing(root_dir_test):
    """
    Processes the dataset for testing by extracting features from each audio file
    in the root directory.

    Parameters:
    - root_dir_test: Directory where the test audio files are located.
    - n_mfcc: Number of Mel-Frequency Cepstral Coefficients to extract.
    """
    features = []
    filenames = []

    for filename in os.listdir(root_dir_test):
        if filename.endswith('.au'):
            file_path = os.path.join(root_dir_test, filename)
            try:
                # Call the feature extraction function
                audio_features = extract_features(file_path)
                features.append(audio_features)
                filenames.append(filename)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Convert list of features to NumPy array
    features = np.array(features)

    feature_columns = ['feature_' + str(i + 1) for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feature_columns)
    # Add filenames as an identification column
    df['id'] = filenames

    name = 'test_features.csv'
    df.to_csv(name, index=False)


process_dataset_for_training('data/train')
process_dataset_for_testing('data/test')
