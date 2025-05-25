import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np


def visualize_vector_sequence_single(
    vectors,
    title="Vector Sequence",
    figsize=(10, 8),
    point_labels=None,
    show_points=True,
    arrow_color="blue",
    point_color="red",
    point_alpha=1.0,
    arrow_alpha=1.0,
    label_alpha=1.0,
):
    """
    Visualize a sequence of vectors as connected arrows in 2D space.

    Parameters:
    -----------
    vectors : list or array-like
        List of vectors (points) where each vector should have at least 2 dimensions.
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    point_labels : list of str, optional
        Labels for each point. If None, numbered indices will be used.
    show_points : bool, optional
        Whether to show the points themselves.
    arrow_color : str, optional
        Color for the arrows.
    point_color : str, optional
        Color for the points.
    point_alpha : float, optional
        Alpha (transparency) for the points (0.0 to 1.0).
    arrow_alpha : float, optional
        Alpha (transparency) for the arrows (0.0 to 1.0).
    label_alpha : float, optional
        Alpha (transparency) for the point labels (0.0 to 1.0).

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Convert input to numpy array for easier handling
    vectors = np.array(vectors)

    if len(vectors) < 1:
        raise ValueError("At least one vector is required")

    # Extract x and y coordinates
    x = vectors[:, 0]
    y = vectors[:, 1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points if requested
    if show_points:
        ax.scatter(x, y, color=point_color, s=100, zorder=3, alpha=point_alpha)

    # Create point labels
    if point_labels is None:
        point_labels = [f"{i}" for i in range(len(vectors))]

    # Add point labels
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(
            point_labels[i],
            (xi, yi),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            alpha=label_alpha,
        )

    # Draw arrows between consecutive points
    for i in range(len(vectors) - 1):
        arrow = FancyArrowPatch(
            (x[i], y[i]),
            (x[i + 1], y[i + 1]),
            arrowstyle="->",
            color=arrow_color,
            mutation_scale=20,
            linewidth=2,
            zorder=2,
            alpha=arrow_alpha,
        )
        ax.add_patch(arrow)

    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)

    # Equal aspect ratio to maintain true vector directions
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust axis limits to include all points with some padding
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = max(x_range, y_range) * 0.1

    ax.set_xlim([x_min - padding, x_max + padding])
    ax.set_ylim([y_min - padding, y_max + padding])

    plt.tight_layout()

    return fig, ax


def visualize_vector_sequence_batch(
    vectors,
    title="Vector Sequences",
    figsize=(10, 8),
    point_labels=None,
    show_points=True,
    arrow_color="blue",
    point_color="red",
    point_alpha=1.0,
    arrow_alpha=1.0,
    label_alpha=1.0,
):
    pass
