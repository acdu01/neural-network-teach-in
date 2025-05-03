import matplotlib.pyplot as plt
import numpy as np

# Layout settings
layer_spacing = 3
neuron_radius = 0.2


def draw_g():
    # Plot sigmoid function
    x = np.linspace(-10, 10, 200)
    y = 1 / (1 + np.exp(-x))

    plt.plot(x, y, label="g(x) = 1 / (1 + e^{-x})")
    plt.title("Sigmoid Function g(x)")
    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_neuron(ax, x, y, bias, label=None):

    # Bias color and circle
    # normalize bias for color
    color = plt.cm.RdBu_r(0.5 + bias / 8)  # pylint: disable=no-member
    circle = plt.Circle((x, y), neuron_radius, color=color, ec="black", lw=1.5)
    ax.add_artist(circle)
    # Bias label
    ax.text(
        x, y, f"{float(bias):.2f}", fontsize=8, ha="center", va="center", color="black"
    )
    # Optional label

    if label:
        ax.text(x, y - 0.4, label, fontsize=7, ha="center")


def draw_connection(ax, x1, y1, x2, y2, weight):
    # normalize weight for color
    norm_w = max(min(weight / 4, 1), -1)
    color = plt.cm.RdBu_r(0.5 + norm_w / 2)  # pylint: disable=no-member
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=abs(weight))
    # Midpoint label
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ax.text(
        mx,
        my,
        f"{weight:.2f}",
        fontsize=6,
        ha="center",
        va="center",
        color="gray",
        rotation=30,
    )


def draw_network(n, weights, biases):
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    layer_positions = []
    for i, num_neurons in enumerate(n):
        y_spacing = 1.5
        y_start = -(num_neurons - 1) * y_spacing / 2
        layer = [
            (i * layer_spacing, y_start + j * y_spacing) for j in range(num_neurons)
        ]
        layer_positions.append(layer)

    # Draw connections with weights
    for l in range(len(weights)):
        for j, (x1, y1) in enumerate(layer_positions[l]):
            for k, (x2, y2) in enumerate(layer_positions[l + 1]):
                draw_connection(ax, x1, y1, x2, y2, weights[l][k, j])

    # Draw neurons with bias labels
    for l, layer in enumerate(layer_positions):
        for i, (x, y) in enumerate(layer):
            if l == 0:
                draw_neuron(ax, x, y, 0, label=f"Input {i+1}")
            else:
                draw_neuron(ax, x, y, biases[l - 1][i], label=f"L{l}N{i+1}")

    plt.title("Neural Network Diagram: Weights & Biases", fontsize=14)
    plt.show()


def plot_cost_surface(
    A0, Y, feed_forward, cost_fn, W1, w1_idx=(0, 0), w2_idx=(1, 0), grid_size=50
):
    w1_vals = np.linspace(-2, 2, grid_size)
    w2_vals = np.linspace(-2, 2, grid_size)
    costs = np.zeros((grid_size, grid_size))
    W1_orig = W1.copy()

    for i, v1 in enumerate(w1_vals):
        for j, v2 in enumerate(w2_vals):
            W1[w1_idx] = v1
            W1[w2_idx] = v2
            y_hat, _ = feed_forward(A0)
            costs[j, i] = cost_fn(y_hat, Y)

    W1[:] = W1_orig

    # Plotting
    W1_grid, W2_grid = np.meshgrid(w1_vals, w2_vals)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(W1_grid, W2_grid, costs, cmap="viridis")
    ax.set_xlabel(f"W1{w1_idx}")
    ax.set_ylabel(f"W1{w2_idx}")
    ax.set_zlabel("Cost")
    ax.set_title("Cost Surface over Two Weights")
    plt.show()


def plot_inputs_by_output(A0, Y):

    # A0 is shape (2, m), so we transpose to (m, 2)
    X = A0.T
    y = Y.flatten()  # shape (m,)

    plt.figure(figsize=(8, 6))
    for label in [0, 1]:
        idx = y == label
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            label=f"Popular = {label}",
            marker="o" if label == 1 else "x",
            c="green" if label == 1 else "red",
            edgecolors="k",
        )

    plt.xlabel("Genre Popularity Index")
    plt.ylabel("Years Since Publication")
    plt.title("Book Inputs Colored by Popularity")
    plt.legend()
    plt.grid(True)
    plt.show()
