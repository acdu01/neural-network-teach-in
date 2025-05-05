import numpy as np
import matplotlib.pyplot as plt
import plot

L = 3
n = [2, 3, 3, 1]


# X: [Genre Popularity Index, Years Since Publication]
X = np.array(
    [
        [0.90, 1],
        [0.75, 3],
        [0.30, 10],
        [0.85, 5],
        [0.60, 2],
        [0.95, 0],
        [0.40, 12],
        [0.70, 4],
        [0.20, 8],
        [0.88, 1],
    ]
)
# y: 1 = popular, 0 = not popular
y = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1])
m = 10
A0 = X.T
Y = y.reshape(n[L], m)


def g(z):
    return 1 / (1 + np.exp(-1 * z))


def cost(y_hat, y):
    """
    y_hat should be a n^L x m matrix
    y should be a n^L x m matrix
    """
    # 1. losses is a n^L x m
    losses = -((y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat))

    m = y_hat.reshape(-1).shape[0]

    # 2. summing across axis = 1 means we sum across rows,
    #   making this a n^L x 1 matrix
    summed_losses = (1 / m) * np.sum(losses, axis=1)

    return np.sum(summed_losses)


def train(alpha=0.1):
    # Re-initialize weights and biases from scratch each time
    W1 = np.random.randn(n[1], n[0])
    W2 = np.random.randn(n[2], n[1])
    W3 = np.random.randn(n[3], n[2])

    b1 = np.random.randn(n[1], 1)
    b2 = np.random.randn(n[2], 1)
    b3 = np.random.randn(n[3], 1)

    costs = []
    epochs = 2000
    for e in range(epochs):
        # FEED FORWARD
        Z1 = W1 @ A0 + b1
        A1 = g(Z1)

        Z2 = W2 @ A1 + b2
        A2 = g(Z2)

        Z3 = W3 @ A2 + b3
        A3 = g(Z3)
        y_hat = A3

        # COST CALCULATION
        error = cost(y_hat, Y)
        costs.append(error)

        # BACKPROP
        dC_dZ3 = (1 / m) * (A3 - Y)
        dC_dW3 = dC_dZ3 @ A2.T
        dC_db3 = np.sum(dC_dZ3, axis=1, keepdims=True)

        dA2_dZ2 = A2 * (1 - A2)
        dC_dZ2 = (W3.T @ dC_dZ3) * dA2_dZ2
        dC_dW2 = dC_dZ2 @ A1.T
        dC_db2 = np.sum(dC_dZ2, axis=1, keepdims=True)

        dA1_dZ1 = A1 * (1 - A1)
        dC_dZ1 = (W2.T @ dC_dZ2) * dA1_dZ1
        dC_dW1 = dC_dZ1 @ A0.T
        dC_db1 = np.sum(dC_dZ1, axis=1, keepdims=True)

        # UPDATE
        W3 -= alpha * dC_dW3
        b3 -= alpha * dC_db3
        W2 -= alpha * dC_dW2
        b2 -= alpha * dC_db2
        W1 -= alpha * dC_dW1
        b1 -= alpha * dC_db1

    return costs
