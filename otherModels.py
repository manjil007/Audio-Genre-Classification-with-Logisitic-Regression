import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression  # Ensure this is your custom LR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def process_and_evaluate_models(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming the last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode labels and scale features
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

    # Models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(learningRate=0.01, epochs=1000, lambda_=0.01),  # Assuming this matches your custom LR's init parameters
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient Boosting Machine": GradientBoostingClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"{name} Accuracy: {accuracy}")


csv_file_path = 'train_features.csv'
process_and_evaluate_models(csv_file_path)
