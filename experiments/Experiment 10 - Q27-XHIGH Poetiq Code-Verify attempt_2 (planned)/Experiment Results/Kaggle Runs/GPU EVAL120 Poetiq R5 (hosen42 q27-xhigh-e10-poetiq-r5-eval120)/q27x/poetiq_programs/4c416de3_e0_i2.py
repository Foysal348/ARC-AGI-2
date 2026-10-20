import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete local corner patterns defined by zero-colored corners.

    Rule observed from the examples:
    1. The background color is the most frequent color.
    2. Ignore color 0 and the background color when determining style.
       - If background is the maximum remaining color: fill a 3x3 diagonal
         ring/cross around the zero corner, plus the diagonal path to the seed.
       - If background is the minimum remaining color: fill a 2x2 block in the
         outside quadrant of the corner, plus the diagonal path to the seed.
       - Otherwise: fill the corner itself and the two outside orthogonal
         cells, plus the diagonal path to the seed.

    A valid anchor is a zero cell with exactly one horizontal zero neighbor
    and exactly one vertical zero neighbor. The interior diagonal direction is
    the diagonal pointing into the quadrant occupied by those two arms.

    A valid seed is the first non-background, non-zero color found 1 or 2
    cells along that interior diagonal, with only background cells in between.
    Existing non-background pixels are never overwritten.
    """
    if grid.size == 0:
        return grid

    orig = np.array(grid, dtype=int, copy=True)

    if orig.ndim != 2:
        return orig

    h, w = orig.shape
    flat = orig.ravel()

    # Determine background as the most frequent color.
    max_val = int(flat.max()) if flat.size else 0
    counts = np.bincount(flat, minlength=max(10, max_val + 1))
    bg = int(np.argmax(counts))

    # Payload colors: non-zero colors that are not the background.
    payload = set(flat.tolist())
    payload.discard(bg)
    payload.discard(0)

    if not payload:
        return orig

    # Choose the completion style by background rank among payload colors.
    if bg == max(payload):
        style = "max"
    elif bg == min(payload):
        style = "min"
    else:
        style = "mid"

    out = orig.copy()

    def write(r: int, c: int, color: int) -> None:
        """Write color unless this cell already contains a payload color."""
        if 0 <= r < h and 0 <= c < w and out[r, c] not in payload:
            out[r, c] = color

    MAX_DIST = 2

    # Find zero corners: exactly one horizontal and one vertical zero neighbor.
    anchors = []
    for r in range(h):
        for c in range(w):
            if orig[r, c] != 0:
                continue

            up = r > 0 and orig[r - 1, c] == 0
            down = r + 1 < h and orig[r + 1, c] == 0
            left = c > 0 and orig[r, c - 1] == 0
            right = c + 1 < w and orig[r, c + 1] == 0

            vert = int(up) + int(down)
            hor = int(left) + int(right)

            if vert == 1 and hor == 1:
                dr = 1 if down else -1
                dc = 1 if right else -1
                anchors.append((r, c, dr, dc))

    def apply_pattern(ar: int, ac: int, dr: int, dc: int,
                      color: int, dist: int) -> None:
        """Apply the style-specific fill for one detected corner/seed."""
        if style == "max":
            # Cross around the zero corner.
            for rr, cc in ((ar - 1, ac), (ar + 1, ac),
                           (ar, ac - 1), (ar, ac + 1)):
                write(rr, cc, color)

            # Diagonal path from the corner to the seed.
            for k in range(1, dist + 1):
                write(ar + k * dr, ac + k * dc, color)

            # Opposite outside diagonal cell.
            write(ar - dr, ac - dc, color)

        elif style == "min":
            # Full 2x2 outside block, including the corner.
            for rr, cc in ((ar, ac),
                           (ar - dr, ac),
                           (ar, ac - dc),
                           (ar - dr, ac - dc)):
                write(rr, cc, color)

            # Diagonal path from the corner to the seed.
            for k in range(1, dist + 1):
                write(ar + k * dr, ac + k * dc, color)

        else:  # mid
            # Corner itself.
            write(ar, ac, color)

            # Two outside orthogonal cells.
            write(ar - dr, ac, color)
            write(ar, ac - dc, color)

            # Diagonal path from the corner to the seed.
            for k in range(1, dist + 1):
                write(ar + k * dr, ac + k * dc, color)

    # Detect seeds and apply fills.
    for ar, ac, dr, dc in anchors:
        seed_color = None
        seed_dist = 0

        for d in range(1, MAX_DIST + 1):
            r = ar + d * dr
            c = ac + d * dc

            if not (0 <= r < h and 0 <= c < w):
                break

            v = orig[r, c]

            if v in payload:
                seed_color = v
                seed_dist = d
                break
            elif v == bg:
                continue
            else:
                # A zero or unrelated payload blocks this diagonal.
                break

        if seed_color is None:
            continue

        apply_pattern(ar, ac, dr, dc, seed_color, seed_dist)

    return out