import numpy as np
import GradientDescent


class LogisticRegression:
    def __init__(self, learningRate=0.01, epochs=1000, lambda_=0.01):
        """
        Initializes the logistic regression model.
        :param learningRate: (alpha) learning rate for gradient descent / controls step size
        :param epochs: number of iterations for gradient descent
        :param lambda_: regularization parameter to prevent overfitting
        """
        self.alpha = learningRate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.weight = None

    def fit(self, input_features, target_labels):
        """
        Fits the logistic regression model to the training data.
        :param input_features:
        :param target_labels:
        :return:
        """
        m, n = input_features.shape
        K = target_labels.shape[1]
        self.weight = np.zeros((n, K))  # Weight initialization

        for epoch in range(self.epochs):
            Z = np.dot(input_features, self.weight)  # Linear predictions
            print("Z = ", Z)
            P = GradientDescent.softmax(Z)  # Softmax probabilities
            print("p = ", P)
            # Gradient computation
            gradient = np.dot(input_features.T, (P - target_labels)) / m + self.lambda_ * self.weight
            print("gradient = ", gradient)
            # Weight update
            self.weight -= self.alpha * gradient
            print("weight = ", self.weight)

    def predict(self, input_features):
        Z = np.dot(input_features, self.weight)
        P = GradientDescent.softmax(Z)
        return np.argmax(P, axis=1)

    def evaluate(self, input_features, target_labels):
        predictions = self.predict(input_features)
        accuracy = np.mean(predictions == np.argmax(target_labels, axis=1))
        return accuracy
