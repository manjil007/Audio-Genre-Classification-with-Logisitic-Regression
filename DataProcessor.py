import os
import numpy as np
import pandas as pd
from test import extract_mfccs, extract_spectral_centroid, extract_spectral_flatness, \
                  extract_chroma_stft, extract_short_time_fourier, extract_zero_crossing
import librosa

def calculate_total_duration(audio_file):
    """
    Calculates the total duration of an audio file.

    Parameters:
    - audio_file: Path to the audio file.

    Returns:
    - duration: Total duration of the audio file in seconds.
    """
    y, sr = librosa.load(audio_file, sr=None)
    return librosa.get_duration(y=y, sr=sr)

def process_dataset(root_dir, segments=10):
    """
    Processes the dataset to extract features and labels, dividing each song into segments.

    Parameters:
    - root_dir: The root directory where the genre directories are located.
    - segments: Number of segments to divide each song into.

    Returns:
    - features: A list of extracted features from all audio segments.
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
                try:
                    total_duration = calculate_total_duration(file_path)
                    segments_features = []
                    for segment_index in range(segments):
                        start_time = segment_index * (total_duration / segments)
                        end_time = (segment_index + 1) * (total_duration / segments)
                        segment_file_path = f"{file_path[:-4]}_{segment_index}.wav"

                        segment_features = []
                        for feature_function in feature_functions.values():
                            extracted_feature = feature_function(segment_file_path)
                            if extracted_feature is not None:
                                segment_features.extend(extracted_feature)
                            else:
                                print(f"Skipping segment: {segment_file_path}")
                                break
                        else:
                            segments_features.append(segment_features)

                    if segments_features:
                        features.append(np.concatenate(segments_features))
                        labels.append(genre)
                except Exception as e:
                    print(f"Error processing file: {file_path}: {e}")

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
features, labels = process_dataset(root_dir, segments=10)
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

labels_column = labels.reshape(-1, 1)

combined_data = np.hstack((features, labels_column))

feature_columns = ['feature_' + str(i + 1) for i in range(features.shape[1])]
df = pd.DataFrame(combined_data, columns=feature_columns + ['Genre'])

df.to_csv('music_genre_dataset.csv', index=False)



