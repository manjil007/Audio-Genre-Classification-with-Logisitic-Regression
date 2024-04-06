import numpy as np


class LogisticRegression:
    def __init__(self, learningRate, epochs, lambda_, regularization='L2'):
        """
        Initializes the logistic regression model.
        :param learningRate: Learning rate for gradient descent / controls step size.
        :param epochs: Number of iterations for gradient descent.
        :param lambda_: Regularization parameter to prevent overfitting.
        :param regularization: Type of regularization ('L1' for Lasso, 'L2' for Ridge).
        """
        self.lr = learningRate
        self.epochs = epochs
        self.reg = lambda_
        self.regularization = regularization
        self.weight = None

    def ReLU(self, X):
        """
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: the value after the ReLU activation is applied to the input (N, layer size)
        """
        out = []
        for v in X:
            out.append([max(0.0, y) for y in v])

        return out

    def ReLU_dev(self, X):
        """
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, layer size)
        :return:
            out: gradient of ReLU given input X
        """
        out = []
        for v in X:
            out.append([0.0 if y < 0 else 1. for y in v])
        return out

    def softmax(self, scores):
        prob = []
        for v in scores:
            # Subtract the max score to prevent overflow
            max_v = np.max(v)
            z = [np.exp(x - max_v) for x in v]
            s = sum(z)
            prob.append([a / s for a in z])
        return prob

    def compute_loss(self, x_pred, y):
        """
        Compute the cross-entropy loss with regularization.
        """
        x_pred = np.clip(x_pred, 1e-12, 1 - 1e-12)
        loss = 0
        for i, v in enumerate(x_pred):
            loss -= np.log(v[y[i].argmax()])

        return loss / y.shape[0]

    def fit(self, input_features, target_labels):
        """
        Fits the logistic regression model to the training data.
        """
        m, n = input_features.shape
        K = target_labels.shape[1]
        self.weight = np.zeros((n, K))
        # self.weight = np.linspace(0, 0.5, n * K).reshape(n, K)

        # Adding bias to input features
        # input_features = np.hstack((np.ones((m, 1)), input_features))
        self.bias = np.ones((1, K))
        for epoch in range(self.epochs):
            Z = np.dot(input_features, self.weight)
            Z = Z + self.bias
            A1 = self.ReLU(Z)
            A2 = self.softmax(A1)

            ds = (np.array(A2) - target_labels) / m
            loss = self.compute_loss(A2, target_labels)
            print(f"Epochs {epoch}, loss : {loss}")
            dz = self.ReLU_dev(Z)
            dl_z = np.multiply(dz, ds)
            gradient = np.dot(input_features.T, dl_z)
            pre_gre = gradient
            # Apply regularization to gradient
            if self.regularization == 'L2':
                gradient = self.reg * self.weight
            elif self.regularization == 'L1':
                gradient = self.reg * np.sign(self.weight)
            self.weight -= self.lr * (gradient + pre_gre)
            self.bias = np.array(np.sum(ds, axis=0))

    def predict(self, input_features):
        m = input_features.shape[0]
        # input_features = np.hstack((np.ones((m, 1)), input_features))  # Add bias term
        Z = np.dot(input_features, self.weight)
        P = self.softmax(Z)
        return np.argmax(P, axis=1)

    def evaluate(self, input_features, target_labels):
        predictions = self.predict(input_features)
        accuracy = np.mean(predictions == np.argmax(target_labels, axis=1))
        return accuracy
