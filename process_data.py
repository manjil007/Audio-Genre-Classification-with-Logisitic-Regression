import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from LogisticRegression import LogisticRegression

# Define a function to extract features from an audio file
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)

    # Add more feature extractions here
    # Example: chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)

    return mfccs_processed  # Modify this if you extract more features

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
                features.append(extract_features(file_path))
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
            # Extract features and append them to the list
            test_features.append(extract_features(file_path))
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
    'audio_file_name': test_filenames,
    'predictions': test_predictions_labels
})

predictions_df.to_csv('test_predictions.csv', index=False)








# At this point, `features` is ready to be used as `input_features`, and `labels_onehot` as `target_labels`
# in your LogisticRegression class.
