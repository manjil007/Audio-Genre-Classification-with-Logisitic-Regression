import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression


# Assuming LogisticRegression is correctly implemented and imported

# Feature extraction function
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    return np.hstack((mfccs_processed, chroma_stft))


# Preparing training data
features = []
labels = []
train_path = 'data/train'
genres = os.listdir(train_path)
for genre in genres:
    genre_path = os.path.join(train_path, genre)
    for filename in os.listdir(genre_path):
        if filename.endswith('.au'):
            file_path = os.path.join(genre_path, filename)
            try:
                features.append(extract_features(file_path))
                labels.append(genre)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

features = np.array(features)
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = OneHotEncoder().fit_transform(labels_encoded.reshape(-1, 1)).toarray()

# Standardizing and applying PCA
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
pca = PCA(n_components=0.90)
features_pca = pca.fit_transform(features_scaled)

# Splitting into training and validation sets
features_train, features_val, labels_train, labels_val = train_test_split(
    features_pca, labels_onehot, test_size=0.2, random_state=42)

# Hyperparameter grid
learning_rates = [0.001, 0.01, 0.1]
epochs_values = [10000]
lambda_values = [0.01, 0.1, 1.0]

best_accuracy = 0
best_params = {'learningRate': None, 'epochs': None, 'lambda_': None}

for lr in learning_rates:
    for epochs in epochs_values:
        for lambda_ in lambda_values:
            model = LogisticRegression(learningRate=lr, epochs=epochs, lambda_=lambda_)
            model.fit(features_train, labels_train)

            # Assuming the LogisticRegression class has an evaluate method
            accuracy = model.evaluate(features_val, labels_val)
            print(f"Accuracy with lr={lr}, epochs={epochs}, lambda={lambda_}: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'learningRate': lr, 'epochs': epochs, 'lambda_': lambda_}
                # Saving the model with the best validation accuracy
                best_model = model

print(f"Best Model Parameters: {best_params}, Validation Accuracy: {best_accuracy}")

# Assuming you also have a prepared test dataset similar to 'features_pca'
# You would need to repeat the feature extraction, scaling, and PCA transformation for your test dataset here

# Make predictions with the best model
# test_predictions = best_model.predict(test_features_pca)  # This requires your test dataset to be prepared
# test_predictions_labels = le.inverse_transform(test_predictions)

# predictions_df = pd.DataFrame({
#     'id': test_filenames,  # Ensure test_filenames is prepared
#     'class': test_predictions_labels
# })
# predictions_df.to_csv('predictions/best_model_predictions.csv', index=False)
