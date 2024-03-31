import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from LogisticRegression import LogisticRegression
from sklearn.decomposition import PCA


# Define a function to extract features from an audio file
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Add more feature extractions here
    # Example: chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)

    return mfccs_processed  # Modify this if you extract more features


def extract_features_fromSTR(file_path, feature, **kwargs):
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
            extracted = special_args_functions[feature](y=audio, **kwargs)
        elif feature in spectral_functions:
            extracted = spectral_functions[feature](y=audio, sr=sr, **kwargs)
        else:
            raise ValueError(f"Feature '{feature}' is not supported.")

    except ValueError as e:
        print("ValueError encountered while parsing file:", file_path)
        print("Error message:", e)
        return None
    except Exception as e:
        print("Error encountered while parsing file:", file_path)
        print("Error message:", e)
        return None

    return np.mean(extracted.T, axis=0)


# Prepare to collect features and labels
features = []
labels = []

# Iterate through each genre folder
train_path = 'data/train'
genres = os.listdir(train_path)
for genre in genres:
    genre_path = os.path.join(train_path, genre)
    for filename in os.listdir(genre_path):
        if filename.endswith('.au'):
            file_path = os.path.join(genre_path, filename)
            try:
                # Extract features and append them to the list, along with the genre label
                mfcc = extract_features(file_path)
                zero_crossing = extract_features_fromSTR(file_path, 'zero_crossing')
                chroma_stft = extract_features_fromSTR(file_path, 'chroma_stft')
                stft = extract_features_fromSTR(file_path, 'short_time_fourier', n_fft=2048, hop_length=512)
                centroid = extract_features_fromSTR(file_path, 'spectral_centroid')
                bandwidth = extract_features_fromSTR(file_path, 'spectral_bandwidth')
                contrast = extract_features_fromSTR(file_path, 'spectral_contrast')
                #flatness = extract_features_fromSTR(file_path, 'spectral_flatness')
                combined_features = np.hstack((mfcc, chroma_stft))

                features.append(combined_features)
                labels.append(genre)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

features = np.array(features)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_encoded = labels_encoded.reshape(len(labels_encoded), 1)
ohe = OneHotEncoder()  # Removing the `sparse=False` argument
labels_onehot = ohe.fit_transform(labels_encoded).toarray()
print(labels_onehot)

# Prepare to collect features for test data
test_features = []
test_filenames = []

# Specify the path to your test data
test_path = 'data/test'

# Iterate through each file in the test data directory
for filename in os.listdir(test_path):
    if filename.endswith('.au'):
        file_path = os.path.join(test_path, filename)
        try:
            test_mfcc = extract_features(file_path)
            test_zero_crossing = extract_features_fromSTR(file_path, 'zero_crossing')
            test_chroma_stft = extract_features_fromSTR(file_path, 'chroma_stft')
            test_centroid = extract_features_fromSTR(file_path, 'spectral_centroid')
            test_bandwidth = extract_features_fromSTR(file_path, 'spectral_bandwidth')
            test_contrast = extract_features_fromSTR(file_path, 'spectral_contrast')

            # Extract features and append them to the list
            combined_test_features = np.hstack((test_mfcc, test_chroma_stft))
            test_features.append(combined_test_features)
            #test_features.append(extract_features(file_path))
            test_filenames.append(filename)  # Store the filename
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Convert test features to NumPy array
test_features = np.array(test_features)

model = LogisticRegression(learningRate=0.01, epochs=1000, lambda_=0.01)

# Fit the model to your training data
model.fit(features, labels_onehot)

# Assuming `model` is your trained LogisticRegression instance
test_predictions = model.predict(test_features)

# Assuming you have a LabelEncoder instance `le` used for the training labels
# If you didn't save this, you'll need to ensure the same transformation is applied as during training
test_predictions_labels = le.inverse_transform(test_predictions)

# Saving predictions to CSV
# Make sure you've captured `test_filenames` when loading test features
predictions_df = pd.DataFrame({
    'id': test_filenames,
    'class': test_predictions_labels
})

predictions_df.to_csv('test_predictions.csv', index=False)

# At this point, `features` is ready to be used as `input_features`, and `labels_onehot` as `target_labels`
# in your LogisticRegression class.
