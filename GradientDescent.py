import numpy as np


def softmax(Z):
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
    # Shift logits by subtracting the max for numerical stability
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    # Normalize to get probabilities
    softmax_probs = e_Z / e_Z.sum(axis=1, keepdims=True)
    return softmax_probs


def compute_loss(X, Y, W):
    """Compute the cross-entropy loss."""
    Z = np.dot(X, W)
    P = softmax(Z)
    return -np.mean(Y * np.log(P + 1e-9))


def gradient_descent(X, Y, alpha, epochs, lambda_):
    m, n = X.shape
    K = Y.shape[1]
    W = np.zeros((n, K))  # Weight initialization

    for epoch in range(epochs):
        Z = np.dot(X, W)  # Linear predictions
        P = softmax(Z)  # Softmax probabilities

        # Gradient computation
        gradient = np.dot(X.T, (P - Y)) / m + lambda_ * W

        # Weight update
        W -= alpha * gradient

        # Print loss every 100 iterations
        if epoch % 100 == 0:
            loss = compute_loss(X, Y, W)
            print(f"Loss at epoch {epoch}: {loss}")

    return W






