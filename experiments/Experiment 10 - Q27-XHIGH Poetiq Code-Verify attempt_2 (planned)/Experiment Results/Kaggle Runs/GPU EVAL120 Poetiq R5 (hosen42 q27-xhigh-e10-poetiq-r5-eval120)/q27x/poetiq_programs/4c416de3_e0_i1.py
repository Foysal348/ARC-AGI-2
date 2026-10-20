import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete diagonal corner patterns.

    The grid has a dominant background color.  Zero cells that are either
    isolated or corners of zero-shaped structures act as anchors.  Colored
    pixels on a diagonal from such an anchor are seeds.  For every anchor and
    diagonal seed direction, the seed color is propagated according to the
    pattern style.

    The style is determined by the background color's rank among non-zero
    non-background colors:
      - background is the maximum non-zero color:
            fill a 3x3 diagonal ring around the anchor:
            the four orthogonal neighbors, the seed diagonal, and the opposite
            diagonal; the anchor remains 0.
      - background is the minimum non-zero color:
            fill the full 2x2 opposite block at the anchor, plus the diagonal
            path from the anchor to the farthest seed.
      - otherwise:
            fill the anchor, the two orthogonal cells in the opposite quadrant,
            and the diagonal path to the farthest seed.

    This reproduces the three example styles.
    """
    if grid.size == 0:
        return grid

    grid = np.array(grid, copy=True)
    h, w = grid.shape

    # Background is the most frequent color.
    bg = Counter(grid.ravel()).most_common(1)[0][0]

    # Non-background payload colors (ignore 0 and background).
    payload_colors = set(grid.ravel().tolist())
    payload_colors.discard(0)
    payload_colors.discard(bg)

    if not payload_colors:
        return grid

    # Choose the completion style.
    if bg == max(payload_colors):
        style = "max"      # 3x3 diagonal ring
    elif bg == min(payload_colors):
        style = "min"      # full 2x2 opposite block + diagonal path
    else:
        style = "mid"      # anchor + opposite orthogonal cells + diagonal path

    def set_if_allowed(r, c, color):
        if 0 <= r < h and 0 <= c < w:
            # Do not overwrite existing non-background payload pixels.
            # Existing 0 / background cells may be rewritten.
            if grid[r, c] not in payload_colors:
                grid[r, c] = color

    # Identify anchor zero cells:
    #   - isolated zero: no orthogonal zero neighbors
    #   - corner zero: exactly two orthogonal zero neighbors, one horizontal
    #     and one vertical.
    anchors = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                continue

            up = r > 0 and grid[r - 1, c] == 0
            down = r + 1 < h and grid[r + 1, c] == 0
            left = c > 0 and grid[r, c - 1] == 0
            right = c + 1 < w and grid[r, c + 1] == 0

            count = int(up) + int(down) + int(left) + int(right)
            vertical = int(up) + int(down)
            horizontal = int(left) + int(right)

            if count == 0:
                anchors.append((r, c))
            elif count == 2 and vertical == 1 and horizontal == 1:
                anchors.append((r, c))

    # Process each anchor in the four diagonal directions.
    for ar, ac in anchors:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            # Find the farthest seed color on this diagonal.
            # We allow background cells along the path, but stop at another
            # payload pixel only after recording it; if the path contains
            # different payload colors, use the nearest/farthest consistent
            # run by taking the first color seen and ignoring mismatches.
            color = None
            far_k = 0

            r, c = ar, ac
            k = 0
            while True:
                r += dr
                c += dc
                if not (0 <= r < h and 0 <= c < w):
                    break
                k += 1

                v = grid[r, c]

                # A background cell is just an empty path cell.
                if v == bg:
                    continue

                # Another zero may be an obstacle beyond the intended pattern.
                if v == 0:
                    break

                # Payload pixel: candidate seed/path cell.
                if v in payload_colors:
                    if color is None:
                        color = v
                        far_k = k
                    elif v == color:
                        far_k = k
                    else:
                        # Mixed colors on one diagonal: stop at first mismatch.
                        break
                    continue

                # Any other unexpected value: stop.
                break

            if color is None or far_k == 0:
                continue

            # Apply the style-specific fill.
            if style == "max":
                # 3x3 diagonal ring around the anchor.
                for rr, cc in (
                    (ar - 1, ac),
                    (ar + 1, ac),
                    (ar, ac - 1),
                    (ar, ac + 1),
                    (ar - dr, ac - dc),       # opposite diagonal
                    (ar + far_k * dr, ac + far_k * dc)  # farthest seed/path
                ):
                    set_if_allowed(rr, cc, color)

                # If there are intermediate diagonal cells, fill them too.
                for t in range(1, far_k + 1):
                    set_if_allowed(ar + t * dr, ac + t * dc, color)

            elif style == "min":
                # Full 2x2 block in the quadrant opposite the seed.
                for rr, cc in (
                    (ar, ac),
                    (ar - dr, ac),
                    (ar, ac - dc),
                    (ar - dr, ac - dc),
                ):
                    set_if_allowed(rr, cc, color)

                # Diagonal path from anchor to the farthest seed.
                for t in range(1, far_k + 1):
                    set_if_allowed(ar + t * dr, ac + t * dc, color)

            else:  # style == "mid"
                # Anchor cell.
                set_if_allowed(ar, ac, color)

                # Two orthogonal cells in the opposite quadrant.
                set_if_allowed(ar - dr, ac, color)
                set_if_allowed(ar, ac - dc, color)

                # Diagonal path to the farthest seed.
                for t in range(1, far_k + 1):
                    set_if_allowed(ar + t * dr, ac + t * dc, color)

    return grid