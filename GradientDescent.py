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


# Example usage:
# Define input features and target classes
X = np.array([[2, 60, 1],  # Added 1 for bias
              [8, 30, 1],  # Added 1 for bias
              [12, 10, 1]])  # Added 1 for bias

Y = np.array([[1, 0, 0],  # One-hot encoding for Bird
              [0, 1, 0],  # One-hot encoding for Cat
              [0, 0, 1]])  # One-hot encoding for Dog

alpha = 0.01  # Learning rate
epochs = 1000  # Number of epochs for gradient descent
lambda_ = 0.01  # Regularization parameter

W = gradient_descent(X, Y, alpha, epochs, lambda_)
print("Optimized weights:", W)


# Optimized weights explanation:
# Optimized weights: [[-0.92529693,  0.09790189,  0.82739504],
#                     [ 0.21098705,  0.08846652, -0.29945357],
#                     [-0.06307854,  0.0082568,   0.05482174]]
# Each row in the weights matrix corresponds to a different feature of the input data,
# and each column corresponds to a different class (Bird, Cat, Dog).

# First Row (Bias Terms):
# - The first row contains the bias weights for each class.
# - The first value (-0.92529693) is the bias for the Bird class. It adjusts the classification threshold
#   independently of the input features, influencing the base likelihood of classifying a sample as a Bird.
# - The second value (0.09790189) is the bias for the Cat class, serving a similar purpose for Cat classification.
# - The third value (0.82739504) is the bias for the Dog class, adjusting the base likelihood of classifying
#   a sample as a Dog.

# Second Row (Weights for the 'Size' Feature):
# - The second row contains the weights associated with the 'Size' feature for each class.
# - The first value (0.21098705) indicates how changes in the 'Size' feature affect the probability
#   of the sample being classified as a Bird. A positive weight suggests that larger values of 'Size'
#   increase the likelihood of the sample being a Bird.
# - The second value (0.08846652) shows the influence of 'Size' on classifying a sample as a Cat.
#   Similarly, a positive value here indicates that larger 'Size' values slightly increase the likelihood
#   of Cat classification.
# - The third value (-0.29945357) demonstrates the impact of 'Size' on Dog classification. A negative weight
#   here suggests that larger 'Size' values decrease the likelihood of the sample being classified as a Dog.

# To use these weights for prediction:
# 1. Compute the weighted sum of the input features and the corresponding weights for each class,
#    including the bias term.
# 2. Apply the softmax function to the weighted sums to obtain the probability distribution across
#    the classes for a given sample.
# 3. The class with the highest probability is the predicted class for the sample.
