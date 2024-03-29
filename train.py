import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from LogisticRegression import LogisticRegression


# Function to read dataset and prepare features and labels
def prepare_data(csv_file):
    df = pd.read_csv(csv_file)
    # Assuming the last column is 'Genre'
    X = df.drop('Genre', axis=1).values  # Features
    y = df['Genre'].values  # Labels

    # Encode the categorical labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Further encode the integer labels to one-hot
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1))

    return X, y_onehot, label_encoder, onehot_encoder


# Load and prepare the training data
train_csv = 'music_genre_dataset.csv'
X_train, y_train, label_encoder, onehot_encoder = prepare_data(train_csv)

# Load and prepare the testing data
test_csv = 'test_music_genre_dataset.csv'
X_test, y_test, _, _ = prepare_data(test_csv)

# Initialize and fit the logistic regression model
model = LogisticRegression(learningRate=0.01, epochs=1000, lambda_=0.01)
model.fit(X_train, y_train)

# Predict on the test set
y_pred_onehot = model.predict(X_test)  # Predicted labels as one-hot encoded
# Convert one-hot encoded predictions back to label indices
y_pred_indices = np.argmax(y_pred_onehot, axis=1)
# Convert label indices back to original labels
y_pred_labels = label_encoder.inverse_transform(y_pred_indices)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Optionally, you might want to see the predicted labels alongside true labels
true_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=1))
comparison = pd.DataFrame({'True Labels': true_labels, 'Predicted Labels': y_pred_labels})
print(comparison)
