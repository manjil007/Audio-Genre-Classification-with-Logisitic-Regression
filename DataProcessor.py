import os
import numpy as np
import pandas as pd
from test import extract_mfccs, extract_spectral_centroid, extract_spectral_flatness, \
                  extract_chroma_stft, extract_short_time_fourier, extract_zero_crossing

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

            if file.endswith('.mp3') or file.endswith('.wav') or file.endswith('.au'):
                file_features = []
                for feature_function in feature_functions.values():
                    extracted_feature = feature_function(file_path)
                    if extracted_feature is not None:
                        file_features.extend(extracted_feature)
                    else:
                        print(f"Skipping file: {file_path}")
                        break
                else:
                    features.append(file_features)
                    labels.append(genre)

    return np.array(features), np.array(labels)

feature_functions = {
    'mfcc': extract_mfccs,
    'spectral_centroid': extract_spectral_centroid,
    'spectral_flatness': extract_spectral_flatness,
    'chroma_stft': extract_chroma_stft,
    'short_time_fourier': extract_short_time_fourier,
    'zero_crossing': extract_zero_crossing
}
root_dir = 'data/train'
features, labels = process_dataset(root_dir)
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

labels_column = labels.reshape(-1, 1)

combined_data = np.hstack((features, labels_column))

feature_columns = ['feature_' + str(i + 1) for i in range(features.shape[1])]
df = pd.DataFrame(combined_data, columns=feature_columns + ['Genre'])

df.to_csv('music_genre_dataset.csv', index=False)

