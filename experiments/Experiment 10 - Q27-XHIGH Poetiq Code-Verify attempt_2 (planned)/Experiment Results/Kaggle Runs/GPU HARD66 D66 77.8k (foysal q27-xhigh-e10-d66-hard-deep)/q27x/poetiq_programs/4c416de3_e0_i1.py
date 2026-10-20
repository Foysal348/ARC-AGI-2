import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    ==============
    The grid has a dominant background color, 0-colored walls, and a few
    non-background, non-zero "source" pixels.

    For each source pixel, look diagonally in the four diagonal directions.
    If the line from the source to a 0-cell is completely empty except for
    background cells, and that 0-cell is a zero corner whose two perpendicular
    zero neighbors point toward the source quadrant, then "project" the source
    color to that corner:

    1. Paint the diagonal cells from the source up to and including the corner.
    2. Paint the two exterior edge-neighbor cells of the corner.
    3. If the source is more than one diagonal step away from the corner,
       also paint the exterior far-diagonal cell one step outside the corner.

    Existing non-background, non-zero pixels are never overwritten.
    """

    if grid.size == 0:
        return grid

    # Identify the background color as the most frequent cell value.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    out = grid.copy()
    h, w = grid.shape

    def safe(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def paint(r: int, c: int, color: int) -> None:
        if not safe(r, c):
            return
        # Only overwrite background or wall cells; preserve other objects.
        if grid[r, c] in (bg, 0):
            out[r, c] = color

    # All possible source cells: neither background nor zero.
    source_mask = (grid != bg) & (grid != 0)
    sources = np.argwhere(source_mask)

    for r, c in sources:
        color = int(grid[r, c])

        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr, cc = r, c
            dist = 0

            # Move diagonally until we hit something non-background.
            while True:
                dist += 1
                nr, nc = rr + dr, cc + dc
                if not safe(nr, nc):
                    break

                val = grid[nr, nc]

                if val == bg:
                    rr, cc = nr, nc
                    continue

                # We require the first non-background cell to be a zero corner.
                if val != 0:
                    break

                # Direction from the corner back to the source quadrant.
                sr, sc = -dr, -dc

                # The zero cell must have the two perpendicular zero neighbors
                # that form a corner opening toward the source.
                corner_ok = False
                if safe(nr + sr, nc) and safe(nr, nc + sc):
                    if grid[nr + sr, nc] == 0 and grid[nr, nc + sc] == 0:
                        corner_ok = True

                if not corner_ok:
                    break

                # Valid zero corner found.
                # Paint intermediate background cells along the diagonal.
                for k in range(1, dist):
                    pr = r + dr * k
                    pc = c + dc * k
                    paint(pr, pc, color)

                # Paint the corner itself.
                paint(nr, nc, color)

                # Paint the two outside edge cells adjacent to the corner.
                paint(nr - sr, nc, color)
                paint(nr, nc - sc, color)

                # For non-immediate corners, also paint the outside far diagonal.
                if dist > 1:
                    paint(nr - sr, nc - sc, color)

                break

    return out