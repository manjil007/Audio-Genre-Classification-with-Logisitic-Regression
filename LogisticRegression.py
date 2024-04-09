import numpy as np

class LogisticRegression:
    def __init__(self, learningRate, epochs, lambda_, regularization='L2', loss_threshold=0.125):
        """
        Initializes the logistic regression model.
        :param learningRate: (alpha) learning rate for gradient descent / controls step size
        :param epochs: number of iterations for gradient descent
        :param lambda_: regularization parameter to prevent overwriting
        """
        self.alpha = learningRate
        self.epochs = epochs
        self.lambda_ = lambda_
        self.weight = None
        self.regularization = regularization
        self.loss_threshold = loss_threshold
        self.loss_history = []

    def softmax(self, Z):
        """
        Compute softmax values for each set of scores in Z.

        The softmax function automatically handles the computation of probabilities
        for all classes, including the last class K, in a unified manner.

        Arguments:
        Z -- A 2D array where each row contains the logits for a single sample
             and each column corresponds to a class. The array includes logits
             for all K classes.

        Returns:
        softmax_probs -- A 2D array where each row contains the softmax probabilities,
                         which represent the predicted probabilities for each class.
                         The function computes these probabilities in such a way that
                         it inherently includes both the first K-1 classes and the last
                         class K without needing explicit separation. The sum of probabilities
                         across all classes for a given sample is guaranteed to be 1.

                         Specifically:
                         - For each of the first K-1 classes, it calculates
                           P(Y = y_k | X) = exp(z_k) / sum(exp(z_j) for j=1 to K),
                           where z_k is the logit for class k.
                         - For the last class K, it inherently calculates its probability
                           as part of the normalization process, ensuring that the
                           probabilities for all classes sum to 1.
        """
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        softmax_probs = e_Z / e_Z.sum(axis=1, keepdims=True)
        return softmax_probs

    def compute_loss(self, X, Y, W):
        """Compute the cross-entropy loss."""
        Z = np.dot(X, W)
        P = self.softmax(Z)
        return -np.mean(Y * np.log(P + 1e-9))

    def fit(self, input_features, target_labels):
        """
        Fits the logistic regression model to the training data.
        :param input_features:
        :param target_labels:
        :return:
        """

        bias = np.ones((input_features.shape[0], 1))
        input_features = np.hstack((bias, input_features))

        m, n = input_features.shape
        K = target_labels.shape[1]
        self.weight = np.zeros((n, K))
        for epoch in range(self.epochs):
            Z = np.dot(input_features, self.weight)
            P = self.softmax(Z)
            target_labels_dense = target_labels.toarray()
            loss = -np.mean(target_labels_dense * np.log(P + 1e-9))
            self.loss_history.append(loss)
            if loss < self.loss_threshold:
                print(f"Stopping early due to loss threshold met: {loss} at epoch {epoch}")
                break
            gradient = np.dot(input_features.T, (P - target_labels)) / m + self.lambda_ * self.weight
            self.weight -= self.alpha * gradient

            if self.regularization == 'L2':
                gradient += (self.lambda_ / m) * self.weight
            elif self.regularization == 'L1':
                gradient += (self.lambda_ / m) * np.sign(self.weight)

            self.weight -= self.alpha * gradient

    def predict(self, input_features):
        bias = np.ones((input_features.shape[0], 1))
        input_features_bias = np.hstack((bias, input_features))
        Z = np.dot(input_features_bias, self.weight)
        P = self.softmax(Z)
        return np.argmax(P, axis=1)

    def evaluate(self, input_features, target_labels):
        predictions = self.predict(input_features)
        accuracy = np.mean(predictions == np.argmax(target_labels, axis=1))
        return accuracy
