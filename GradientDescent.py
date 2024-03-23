import numpy as np


def softmax(Z):
    # Compute the softmax of vector Z in a numerically stable way
    exp_shifted = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def compute_cost(X, Y, W, lambda_):
    # Compute the cost function (cross-entropy loss with L2 regularization)
    m = X.shape[0]
    Z = np.dot(X, W)
    A = softmax(Z)
    cost = (-1 / m) * np.sum(Y * np.log(A + 1e-15))  # Adding epsilon for numerical stability
    reg_cost = (lambda_ / (2 * m)) * np.sum(W ** 2)
    return cost + reg_cost


def gradient_descent(X, Y, alpha, lambda_, iterations):
    m, n = X.shape
    K = Y.shape[1]
    W = np.zeros((n, K - 1))  # Initialize weights
    cost_history = []

    for i in range(iterations):
        Z = np.dot(X, W)
        A = softmax(Z)

        # Gradient computation with L2 regularization
        gradient = (1 / m) * np.dot(X.T, (A[:, :-1] - Y[:, :-1])) + (lambda_ / m) * W

        # Update weights
        W = W - alpha * gradient

        # Compute and record the cost
        cost = compute_cost(X, Y, W, lambda_)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return W, cost_history


# Example usage:
# Assume X_train is your input features matrix of shape (m, n) and Y_train is one-hot encoded target matrix of shape (m, K)
# Remember to add a column of ones to X_train for the bias term
# X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
alpha = 0.01  # Learning rate
lambda_ = 0.1  # Regularization strength
iterations = 1000  # Number of iterations

# You should convert your Y_train to one-hot encoded format if it's not already
# Example: Y_train_encoded = one_hot_encode(Y_train, num_classes=K)

# Perform gradient descent
# W_optimized, cost_history = gradient_descent(X_train, Y_train_encoded, alpha, lambda_, iterations)

# W_optimized contains the optimized weights, cost_history contains the cost function value at each iteration






