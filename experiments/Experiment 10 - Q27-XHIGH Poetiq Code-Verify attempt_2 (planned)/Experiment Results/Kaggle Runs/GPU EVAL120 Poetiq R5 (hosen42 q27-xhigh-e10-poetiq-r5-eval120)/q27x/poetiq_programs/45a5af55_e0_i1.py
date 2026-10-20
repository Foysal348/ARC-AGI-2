import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform a vertical color sequence into a square of nested rectangular frames.

    The input is interpreted as a 1D sequence of row colors. Consecutive rows with
    the same color form one "layer" whose thickness is the number of consecutive
    rows. The output is a square constructed from the outside inward:
      - each non-final layer becomes a rectangular frame of its color and thickness;
      - the final layer becomes the central filled square.

    The output size is:
        2 * sum(thicknesses of all layers except the last) + thickness(last layer)

    For robustness, if a row is not perfectly uniform, the most frequent color
    in that row is used.
    """
    if grid is None or grid.size == 0:
        return np.zeros_like(grid)

    grid = np.asarray(grid)
    H, W = grid.shape

    # Extract the representative color of each row.
    row_colors = []
    for r in range(H):
        row = grid[r]
        if W == 0:
            row_colors.append(0)
            continue
        if len(np.unique(row)) == 1:
            row_colors.append(int(row[0]))
        else:
            # Fallback: use the most frequent color in the row.
            vals, counts = np.unique(row, return_counts=True)
            row_colors.append(int(vals[np.argmax(counts)]))

    # Group consecutive identical row colors into layers (color, thickness).
    layers = []
    for c in row_colors:
        if layers and layers[-1][0] == c:
            layers[-1][1] += 1
        else:
            layers.append([c, 1])

    if not layers:
        return np.zeros((0, 0), dtype=grid.dtype)

    last_color, last_thickness = layers[-1]
    outer_thickness_sum = sum(t for _, t in layers[:-1])
    N = 2 * outer_thickness_sum + last_thickness

    out = np.zeros((N, N), dtype=grid.dtype)

    offset = 0
    # Draw each non-final layer as a frame of the given thickness.
    for color, thickness in layers[:-1]:
        lo = offset
        hi = N - offset

        # Top and bottom borders.
        out[lo:lo + thickness, lo:hi] = color
        out[hi - thickness:hi, lo:hi] = color

        # Left and right borders.
        out[lo:hi, lo:lo + thickness] = color
        out[lo:hi, hi - thickness:hi] = color

        offset += thickness

    # The final layer is the central filled square.
    out[offset:offset + last_thickness, offset:offset + last_thickness] = last_color

    return out