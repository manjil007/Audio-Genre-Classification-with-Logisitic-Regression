import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression
# Assuming LogisticRegression is correctly implemented and imported


def extract_features(file_path, n_mfcc):
    """
    Extracts features from an audio file without dividing it into segments.

    Parameters:
    - file_path: Path to the audio file.
    - n_mfcc: Number of MFCCs to extract.

    Returns:
    - features: Extracted features from the entire audio file.
    """
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Extract MFCCs from the entire audio file
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)

    # Extract Chroma STFT from the entire audio file
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)

    # Combine extracted features
    features = np.hstack((mfccs, chroma_stft))

    return features


# Load pre-extracted features and labels from CSV
df = pd.read_csv('5segments_40mfcc_features.csv')
# Assuming your CSV has columns for features named as 'feature_1', 'feature_2', ..., 'feature_N'
# and a column for labels named 'label'

# Extract features and labels
features = df.drop('Genre', axis=1).values
labels = df['Genre'].values

# Label encoding and one-hot encoding (unchanged)
features = np.array(features)
# Label encoding
le = LabelEncoder()  # Create a LabelEncoder instance
labels_encoded = le.fit_transform(labels)  # Fit and transform the labels to encode them as integers
labels_encoded = labels_encoded.reshape(-1, 1)
ohe = OneHotEncoder()
labels_onehot = ohe.fit_transform(labels_encoded).toarray()

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA(n_components=0.90)
features_pca = pca.fit_transform(features_scaled)
print(f"Features before PCA: {features.shape[1]}, after PCA: {features_pca.shape[1]}")

# Prepare and standardize test data (updated to include PCA)
test_features = []
test_filenames = []
test_path = 'data/test'
for filename in os.listdir(test_path):
    if filename.endswith('.au'):
        file_path = os.path.join(test_path, filename)
        try:
            extracted_feature = extract_features(file_path, 40)
            test_features.append(extracted_feature)
            test_filenames.append(filename)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

test_features = np.array(test_features)
test_features_scaled = scaler.transform(test_features)  # Use the same scaler as for the training data
test_features_pca = pca.transform(test_features_scaled)  # Apply the same PCA transformation

# # Logistic regression model initialization and training (unchanged)
# model = LogisticRegression(learningRate=0.01, epochs=1000, lambda_=0.01)
# model.fit(features_pca, labels_onehot)
#
# # Prediction on test data (updated to use PCA-transformed features)
# test_predictions = model.predict(test_features_pca)
# test_predictions_labels = le.inverse_transform(test_predictions)
#
# # Saving predictions to CSV (unchanged)
# predictions_df = pd.DataFrame({
#     'id': test_filenames,
#     'class': test_predictions_labels
# })
# predictions_df.to_csv('test_predictions.csv', index=False)

# Define your parameter grid
learning_rates = [0.1]
epochs_values = [1000]
lambda_values = [0.01]

# Ensure the 'predictions' directory exists
output_dir = 'predictions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for lr in learning_rates:
    for epochs in epochs_values:
        for lambda_ in lambda_values:
            # Initialize and train the logistic regression model
            model = LogisticRegression(learningRate=lr, epochs=epochs, lambda_=lambda_)
            model.fit(features_pca, labels_onehot)

            # Make predictions on the test set
            test_predictions = model.predict(test_features_pca)
            test_predictions_labels = le.inverse_transform(test_predictions)

            # Prepare the DataFrame for saving
            predictions_df = pd.DataFrame({
                'id': test_filenames,
                'class': test_predictions_labels
            })

            # Define the output filename
            filename = f"test_predictions_{lr}_{epochs}_{lambda_}.csv"
            filepath = os.path.join(output_dir, filename)

            # Save predictions to CSV
            predictions_df.to_csv(filepath, index=False)
            print(f"Saved predictions to {filepath}")