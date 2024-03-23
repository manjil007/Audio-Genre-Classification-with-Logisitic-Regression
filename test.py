import os
import numpy as np
import pandas as pd
import librosa


def extract_mfccs(file_path, n_mfcc=13):
    """
    Extracts MFCCs from an audio file.

    Parameters:
    - file_path: Path to the audio file.
    - n_mfcc: Number of MFCCs to extract.

    Returns:
    - A numpy array containing the MFCCs.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_path)
        return None
    return mfccs_processed


def extract_spectral_centroid(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        cents = librosa.feature.spectral_centroid(y=audio, sr=sr)

        cents_processed = np.mean(cents.T, axis=0)
    except:
        print("Error encountered while parsing file: ", file_path)
        return None
    return cents_processed


def extract_spectral_flatness(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        spectral_flatness = librosa.feature.spectral_flatness(audio=audio)

        flatness_processed = np.mean(spectral_flatness.T, axis=0)
    except:
        print("Error encountered while parsing file: ", file_path)
        return None
    return flatness_processed


def extract_chroma_stft(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        chroma_stft = librosa.feature.chroma_stft(audio=audio)

        cstft_processed = np.mean(chroma_stft.T, axis=0)
    except:
        print("Error encountered while parsing file: ", file_path)
        return None
    return cstft_processed


def extract_short_time_fourier(file_path, n_fft=2048, hop_length=512):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        short_time_fourier = librosa.stft(audio=audio, n_fft=n_fft, hop_length=hop_length)
        stft_processed = np.abs(short_time_fourier)
    except:
        print("Error encountered while parsing file:", file_path)
        return None
    return stft_processed


def extract_zero_crossing(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        zero_crossing = librosa.feature.zero_crossings(audio=audio)
        zcross_processed = np.mean(zero_crossing.T, axis=0)
    except:
        print("Error encountered while parsing file:", file_path)
        return None
    return zcross_processed


def extract_features(file_path, feature, **kwargs):
    spectral_functions = {
        'spectral_centroid': librosa.feature.spectral_centroid,
        'spectral_bandwidth': librosa.feature.spectral_bandwidth,
        'spectral_contrast': librosa.feature.spectral_contrast,
        'spectral_flatness': librosa.feature.spectral_flatness,
        # Add more features as needed
    }
    special_args_functions = {
        # these ones only need audio as the first argument and not sr
        'chroma_stft': librosa.feature.chroma_stft,
        'zero_crossing': librosa.feature.zero_crossing_rate,
        'short_time_fourier': librosa.stft,
        # add more features as needed
    }

    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        if feature in special_args_functions:
            extracted = spectral_functions[feature](y=audio, **kwargs)
        elif feature in spectral_functions:
            extracted = spectral_functions[feature](y=audio, sr=sr, **kwargs)
        else:
            raise ValueError(f"Feature '{feature}' is not supported.")

    except:
        print("Error encountered while parsing file:", file_path)
        return None

    return np.mean(extracted.T, axis=0)


def process_dataset(root_dir):
    """
    Processes the dataset to extract features and labels.

    Parameters:
    - root_dir: The root directory where the genre directories are located.

    Returns:
    - features: A list of extracted features from all audio files.
    - labels: A list of genre labels corresponding to each feature set.
    """
    genres = os.listdir(root_dir)
    features = []
    labels = []

    for genre in genres:
        print(f"Processing genre: {genre}")
        genre_dir = os.path.join(root_dir, genre)
        for file in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, file)
            # Ensure we're only processing audio files (adjust the condition as needed)
            if file.endswith('.mp3') or file.endswith('.wav') or file.endswith('.au'):
                mfccs = extract_mfccs(file_path)
                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(genre)

    return np.array(features), np.array(labels)


root_dir = 'data/train'

features, labels = process_dataset(root_dir)
print("feature = ", len(features))
print("labels = ", labels)

# Convert labels to a column vector for concatenation
labels_column = labels.reshape(-1, 1)

# If features is a NumPy array and labels is a list or array, combine them
# Note: This assumes features is a 2D array where rows correspond to samples and columns to features
combined_data = np.hstack((features, labels_column))

# Convert the combined data to a pandas DataFrame
# Generate feature column names based on the number of features
feature_columns = ['mfcc_' + str(i + 1) for i in range(features.shape[1])]
df = pd.DataFrame(combined_data, columns=feature_columns + ['Genre'])

# Write the DataFrame to a CSV file
df.to_csv('music_genre_dataset.csv', index=False)
