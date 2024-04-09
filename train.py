import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from LogisticRegression import LogisticRegression
from sklearn.model_selection import KFold

# Load pre-extracted features from CSV
train_df = pd.read_csv('processed_data/train_features_not22500.csv')  # Adjust this path
test_df = pd.read_csv('venv/test_features_not22500.csv')  # Adjust this path

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
kf = KFold(n_splits=3, shuffle=True, random_state=42)
cv_accuracies = []

evaluation_data = []

for train_index, test_index in kf.split(X_train_pca, y_train_encoded):
    X_cv_train, X_cv_test = X_train_pca[train_index], X_train_pca[test_index]
    y_cv_train, y_cv_test = y_train_onehot[train_index], y_train_onehot[test_index]

    model_cv = LogisticRegression(learningRate=0.1, epochs=10000, lambda_= 0.1, regularization='L2')
    model_cv.fit(X_cv_train, y_cv_train)

    predictions = model_cv.predict(X_cv_test)
    actual_labels = np.argmax(y_cv_test, axis=1)
    correct_predictions = (actual_labels == predictions)

    actual_labels = np.array(actual_labels).flatten()
    predictions = np.array(predictions).flatten()

    for actual, predicted in zip(actual_labels, predictions):
        actual_value = label_encoder.inverse_transform([actual])[0]
        predicted_value = label_encoder.inverse_transform([predicted])[0]
        correct_prediction = actual == predicted

        evaluation_data.append({
            'actual_value': actual_value,
            'predicted_value': predicted_value,
            'Correct_Prediction': correct_prediction
        })

evaluation_df = pd.DataFrame(evaluation_data)
evaluation_df.to_csv("evaluation_results.csv", index=False)

true_predictions_count = evaluation_df['Correct_Prediction'].sum()
total_predictions = len(evaluation_df)

average_accuracy = true_predictions_count / total_predictions
print(f'Average CV Accuracy: {average_accuracy}')

model = LogisticRegression(learningRate=0.1, epochs=1000, lambda_=0.1)
model.fit(X_train_pca, y_train_onehot)

test_predictions = model.predict(X_test_pca)
test_predicted_labels = label_encoder.inverse_transform(test_predictions)

predictions_df = pd.DataFrame({'id': test_ids, 'predicted_genre': test_predicted_labels})
predictions_output_path = 'test_predictions_L2.csv'
predictions_df.to_csv(predictions_output_path, index=False)

print(f'Test predictions saved to {predictions_output_path}')
