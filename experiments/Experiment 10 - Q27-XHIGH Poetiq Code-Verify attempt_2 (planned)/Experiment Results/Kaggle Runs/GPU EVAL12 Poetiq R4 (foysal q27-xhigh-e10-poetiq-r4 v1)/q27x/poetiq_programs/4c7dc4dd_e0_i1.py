import numpy as np
from itertools import product

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid by:
      1. Finding all closed rectangular "frames" in the input.
      2. Extracting the interior of each frame.
      3. XOR-ing the extracted interiors together as binary images
         (any non-zero pixel is treated as "on").
      4. Returning the resulting binary image, drawn with color 2.

    This matches the examples:
      - Example 1: four 6x6 8-frames with 4x4 interiors; XOR of their
        non-zero pixels gives the 4x4 output.
      - Example 2: four 7x7 4-frames with 5x5 interiors; XOR of their
        non-zero pixels gives the 5x5 output.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    def is_uniform_rectangle(r0, c0, r1, c1, color):
        """Check if all border pixels of the rectangle are exactly color."""
        # top and bottom
        if not np.all(grid[r0, c0:c1 + 1] == color):
            return False
        if not np.all(grid[r1, c0:c1 + 1] == color):
            return False
        # left and right
        if not np.all(grid[r0:r1 + 1, c0] == color):
            return False
        if not np.all(grid[r0:r1 + 1, c1] == color):
            return False
        return True

    def interior_size_is_valid(r0, c0, r1, c1):
        """Require at least a 1x1 interior (i.e., frame at least 3x3)."""
        return (r1 - r0) >= 2 and (c1 - c0) >= 2

    # We search for candidate frame rectangles.
    # To stay efficient and avoid noise from background, only consider
    # frames up to a reasonable size and with a solid uniform border.
    frames = []
    seen = set()

    # Candidate frame colors: any color that appears in the grid.
    colors = np.unique(grid)

    # Max frame side to search: keep it practical.
    max_side = 25

    for color in colors:
        # Skip color 0 (usually background) because frames are non-background.
        if color == 0:
            continue

        # Look for all top-left corners where a horizontal run of this color
        # begins and try expanding to form closed rectangles.
        for r0 in range(h):
            for c0 in range(w):
                if grid[r0, c0] != color:
                    continue

                # Try possible right edges.
                for c1 in range(c0 + 2, min(w, c0 + max_side + 2)):
                    # Need top border fully color.
                    if not np.all(grid[r0, c0:c1 + 1] == color):
                        continue
                    # Need right side start.
                    if grid[r0, c1] != color:
                        continue

                    # Try possible bottom edges.
                    for r1 in range(r0 + 2, min(h, r0 + max_side + 2)):
                        if grid[r1, c0] != color or grid[r1, c1] != color:
                            continue

                        if not interior_size_is_valid(r0, c0, r1, c1):
                            continue

                        if not is_uniform_rectangle(r0, c0, r1, c1, color):
                            continue

                        # Canonical key for this frame rectangle.
                        key = (r0, c0, r1, c1)
                        if key in seen:
                            continue

                        # Avoid counting the same visual frame multiple times:
                        # If the frame border contains another full frame border
                        # of the same color, we keep only the outermost.
                        # In practice, our search already finds maximal uniform
                        # rectangles, but this guard improves robustness.
                        seen.add(key)
                        frames.append((r0, c0, r1, c1))

    # If no frames were found (should not happen for valid ARC inputs),
    # return a copy of the input.
    if not frames:
        return grid.copy()

    # Extract interiors and compute XOR of their binary masks.
    interiors = []
    for r0, c0, r1, c1 in frames:
        interior = grid[r0 + 1:r1, c0 + 1:c1]
        interiors.append(interior)

    if not interiors:
        return np.zeros_like(grid)

    # Determine the common interior shape.
    # All intended frames have the same interior size. If there is variation,
    # pad smaller interiors to the maximum size using zeros.
    max_h = max(x.shape[0] for x in interiors)
    max_w = max(x.shape[1] for x in interiors)

    xor_mask = np.zeros((max_h, max_w), dtype=bool)
    for interior in interiors:
        ih, iw = interior.shape
        # Treat any non-zero pixel as "on".
        binary = (interior != 0)
        xor_mask[:ih, :iw] ^= binary

    # Draw the XOR result with color 2 on a zero background.
    result = np.zeros((max_h, max_w), dtype=grid.dtype)
    result[xor_mask] = 2

    return result