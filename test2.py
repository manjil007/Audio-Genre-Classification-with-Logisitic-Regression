import os
import numpy as np
import pandas as pd
import librosa

def extract_feature(file_path, feature_name):
    """
    Extracts specified feature from an audio file.

    Parameters:
    - file_path: Path to the audio file.
    - feature_name: Name of the feature to extract.

    Returns:
    - Processed feature as a numpy array.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        if feature_name == 'mfccs':
            feature = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        elif feature_name == 'spectral_centroid':
            feature = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        elif feature_name == 'spectral_flatness':
            feature = librosa.feature.spectral_flatness(y=audio)
        elif feature_name == 'chroma_stft':
            feature = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        elif feature_name == 'zero_crossing':
            feature = librosa.feature.zero_crossing_rate(y=audio)
        # Add more elif blocks for other features as needed
        else:
            return None
        feature_processed = np.mean(feature.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}, {e}")
        return None
    return feature_processed

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
    all_features = []
    labels = []

    feature_names = ['mfccs', 'spectral_centroid', 'spectral_flatness', 'chroma_stft', 'zero_crossing']  # Add other features names as needed

    for genre in genres:
        print(f"Processing genre: {genre}")
        genre_dir = os.path.join(root_dir, genre)
        for file in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, file)
            if file.endswith('.mp3') or file.endswith('.wav') or file.endswith('.au'):
                features = [extract_feature(file_path, feature) for feature in feature_names]
                # Remove None values and flatten the list
                features = [f for f in features if f is not None]
                if features:
                    combined_features = np.hstack(features)
                    all_features.append(combined_features)
                    labels.append(genre)

    return np.array(all_features), np.array(labels)


root_dir = 'data/train'

# Assuming 'root_dir' is defined and pointing to your dataset directory
features, labels = process_dataset(root_dir)

# Following steps to create DataFrame and CSV file remain the same as in your provided script
print("feature = ", features)
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