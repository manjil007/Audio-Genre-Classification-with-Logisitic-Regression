import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def extract_features(file_path, n_mfcc):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', sr=25000)
    stft = np.abs(librosa.stft(audio))
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio).T, axis=0)
    energy = np.mean(librosa.feature.rms(y=audio).T, axis=0)
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    harmonic = np.mean(librosa.feature.rms(y=y_harmonic).T, axis=0)
    percussive = np.mean(librosa.feature.rms(y=y_percussive).T, axis=0)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).T, axis=0)
    features = np.hstack((mfccs, chroma, spectral_contrast, zero_crossing_rate, energy, harmonic, percussive, tempo, spectral_rolloff))
    return features


def process_and_train_model(train_dir, test_dir, n_mfcc=3, output_dir='predictions'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_features, train_labels = [], []
    for genre in os.listdir(train_dir):
        genre_path = os.path.join(train_dir, genre)
        for filename in os.listdir(genre_path):
            if filename.endswith('.au'):
                file_path = os.path.join(genre_path, filename)
                features = extract_features(file_path, n_mfcc)
                train_features.append(features)
                train_labels.append(genre)

    le = LabelEncoder()
    labels_encoded = le.fit_transform(train_labels)
    ohe = OneHotEncoder()
    labels_onehot = ohe.fit_transform(labels_encoded.reshape(-1, 1)).toarray()

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(np.array(train_features))

    pca = PCA(n_components=0.95)
    train_features_pca = pca.fit_transform(train_features_scaled)

    # KFold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in kf.split(train_features_pca, labels_encoded):
        X_train_cv, X_test_cv = train_features_pca[train_idx], train_features_pca[test_idx]
        y_train_cv, y_test_cv = labels_onehot[train_idx], labels_onehot[test_idx]

        model_cv = LogisticRegression(learningRate=0.1, epochs=1000, lambda_=0.01)
        model_cv.fit(X_train_cv, y_train_cv)
        predictions_cv = model_cv.predict(X_test_cv)
        acc = accuracy_score(np.argmax(y_test_cv, axis=1), predictions_cv)
        accuracies.append(acc)
    print(f"Average Cross-Validation Accuracy: {np.mean(accuracies)}")

    # Retrain model on the full training set
    model = LogisticRegression(learningRate=0.01, epochs=1000, lambda_=0.01)
    model.fit(train_features_pca, labels_onehot)

    # Process and predict on the test set
    test_features, test_filenames = [], []
    for filename in os.listdir(test_dir):
        if filename.endswith('.au'):
            file_path = os.path.join(test_dir, filename)
            features = extract_features(file_path, n_mfcc)
            test_features.append(features)
            test_filenames.append(filename)

    test_features_scaled = scaler.transform(np.array(test_features))
    test_features_pca = pca.transform(test_features_scaled)
    test_predictions = model.predict(test_features_pca)
    test_predictions_labels = le.inverse_transform(test_predictions)

    # Save predictions to CSV
    predictions_df = pd.DataFrame({'id': test_filenames, 'class': test_predictions_labels})
    predictions_df.to_csv(os.path.join(output_dir, 'test_predictions_L2.csv'), index=False)


# Example usage
process_and_train_model('data/train', 'data/test', n_mfcc=20, output_dir='predictions')



