import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression  # Make sure this is your custom implementation
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load pre-extracted features from CSV
train_df = pd.read_csv('train_features.csv')  # Adjust this path
test_df = pd.read_csv('test_features.csv')  # Adjust this path

# Training data
X_train = train_df.drop(['Genre'], axis=1).values
y_train = train_df['Genre'].values

# Test data
X_test = test_df.drop(['id'], axis=1).values
test_ids = test_df['id'].values

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# OneHot encode the labels
onehot_encoder = OneHotEncoder()
y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1))

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Prepare cross-validation
kf =KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

# Perform cross-validation
for train_index, test_index in kf.split(X_train_pca, y_train_encoded):
    X_cv_train, X_cv_test = X_train_pca[train_index], X_train_pca[test_index]
    y_cv_train, y_cv_test = y_train_onehot[train_index], y_train_onehot[test_index]

    # Initialize and train the custom Logistic Regression model
    model_cv = LogisticRegression(learningRate=0.45, epochs=10, lambda_=0.0000001, regularization='L2')
    model_cv.fit(X_cv_train, y_cv_train)

    # Predict and evaluate accuracy
    predictions = model_cv.predict(X_cv_test)
    actual = np.argmax(y_cv_test, axis=1).flatten()
    actual = np.squeeze(np.asarray(actual))
    acc = accuracy_score(actual, predictions)
    cv_accuracies.append(acc)

# Print the average cross-validation accuracy
print(f'Average CV Accuracy: {np.mean(cv_accuracies)}')

# model = LogisticRegression(learningRate=0.1, epochs=1000, lambda_=0.01)
# model.fit(X_train_pca, y_train_onehot)
#
# # Predict on test set
# test_predictions = model.predict(X_test_pca)
# # Convert predictions from one-hot back to original labels
# test_predicted_labels = label_encoder.inverse_transform(test_predictions)

# Save test predictions
# predictions_df = pd.DataFrame({'id': test_ids, 'predicted_genre': test_predicted_labels})
# predictions_output_path = 'test_predictions_L2.csv'
# predictions_df.to_csv(predictions_output_path, index=False)
#
# print(f'Test predictions saved to {predictions_output_path}')
