import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid according to the observed ARC rule:

    1. Red pixels (2) are edge markers.
       - If there is one marker, it indicates one active edge side.
       - If there are two markers, they are on perpendicular edges and define
         an L-shaped cross: a vertical line through the top/bottom marker
         and a horizontal line through the left/right marker.

    2. These lines partition the grid into quadrants.  For each quadrant,
       determine the grid corner it would map to:
           top/bottom side from above/below the horizontal partition
           left/right side from left/right of the vertical partition
       The quadrant is kept only if at least one of its two corner sides is
       indicated by a red marker.

    3. For each kept quadrant:
         - collect all orange (7) pixels in that quadrant;
         - take their bounding box;
         - translate that bounding box to the corresponding grid corner,
           preserving orientation.

    4. Draw red (2) trails along the marker lines:
         - for single top/left markers, extend to the farthest orange pixel
           on that line;
         - for single bottom/right markers, extend to the nearest contiguous
           orange segment on that line;
         - for two-marker L cases, extend each arm to the farthest orange
           pixel on that line.
       Orange pixels already on a drawn line are preserved.

    The background is gray (1).
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = np.ones_like(grid, dtype=int)

    # Locate red markers.
    markers = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 2:
                side = None
                if r == 0:
                    side = "top"
                elif r == H - 1:
                    side = "bottom"
                elif c == 0:
                    side = "left"
                elif c == W - 1:
                    side = "right"
                else:
                    # Fallback: use nearest edge.
                    d = [r, H - 1 - r, c, W - 1 - c]
                    side = ("top", "bottom", "left", "right")[int(np.argmin(d))]
                markers.append((r, c, side))

    if not markers:
        return out

    marker_sides = {m[2] for m in markers}

    # Determine vertical and horizontal partition lines.
    # Vertical line is associated with top/bottom markers.
    # Horizontal line is associated with left/right markers.
    vertical_col = None
    horizontal_row = None

    for _, c, side in markers:
        if side in ("top", "bottom") and vertical_col is None:
            vertical_col = c
    for r, _, side in markers:
        if side in ("left", "right") and horizontal_row is None:
            horizontal_row = r

    # If only a top/bottom marker exists, use the opposite edge as the
    # horizontal partition so the whole shape lies on the marker side.
    if vertical_col is not None and horizontal_row is None:
        # If the marker is on top, active side is top; put partition at bottom.
        # If the marker is on bottom, active side is bottom; put partition at top.
        if any(s == "top" for s in marker_sides):
            horizontal_row = H - 1
        else:
            horizontal_row = 0

    # If only a left/right marker exists, use the opposite edge as the
    # vertical partition.
    if horizontal_row is not None and vertical_col is None:
        if any(s == "left" for s in marker_sides):
            vertical_col = W - 1
        else:
            vertical_col = 0

    # If still missing, fall back to center.
    if vertical_col is None:
        vertical_col = W // 2
    if horizontal_row is None:
        horizontal_row = H // 2

    hr = horizontal_row
    vc = vertical_col

    # Helper: map a quadrant side pair to the active corner.
    def quadrant_side(r, c):
        rs = "top" if r < hr else "bottom"
        cs = "left" if c < vc else "right"
        return rs, cs

    def corner_is_active(rs, cs):
        return rs in marker_sides or cs in marker_sides

    # Collect 7 pixels by active corner.
    corner_pixels = {
        "TL": [],
        "TR": [],
        "BL": [],
        "BR": [],
    }

    for r in range(H):
        if r == hr:
            continue
        for c in range(W):
            if c == vc:
                continue
            if grid[r, c] != 7:
                continue
            rs, cs = quadrant_side(r, c)
            if not corner_is_active(rs, cs):
                continue
            key = "T" + ("L" if cs == "left" else "R") if rs == "top" else \
                  "B" + ("L" if cs == "left" else "R")
            corner_pixels[key].append((r, c))

    def place_corner(pts, corner):
        if not pts:
            return
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        min_r, max_r = min(rs), max(rs)
        min_c, max_c = min(cs), max(cs)

        if corner == "TL":
            dr = 0 - min_r
            dc = 0 - min_c
        elif corner == "TR":
            dr = 0 - min_r
            dc = (W - 1) - max_c
        elif corner == "BL":
            dr = (H - 1) - max_r
            dc = 0 - min_c
        else:  # BR
            dr = (H - 1) - max_r
            dc = (W - 1) - max_c

        for r, c in pts:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                out[nr, nc] = 7

    place_corner(corner_pixels["TL"], "TL")
    place_corner(corner_pixels["TR"], "TR")
    place_corner(corner_pixels["BL"], "BL")
    place_corner(corner_pixels["BR"], "BR")

    # Draw red trails along marker lines.
    def fill_vertical_line(col, marker_row, side):
        if col is None or not (0 <= col < W):
            return
        # Direction into the grid.
        if side == "top":
            step = 1
        elif side == "bottom":
            step = -1
        else:
            step = 1

        # Find all original 7s on this column.
        col_7s = [r for r in range(H) if grid[r, col] == 7]
        if not col_7s:
            # If there is no orange on the line, just mark the marker.
            if 0 <= marker_row < H:
                out[marker_row, col] = 2
            return

        if len(markers) == 1 and side in ("bottom", "right"):
            # Single bottom/right: stop at the nearest contiguous orange segment.
            # Determine the segment containing the orange pixel closest to marker.
            nearest = min(col_7s, key=lambda r: abs(r - marker_row))
            # Expand contiguous segment around nearest.
            lo = nearest
            hi = nearest
            while lo - 1 >= 0 and grid[lo - 1, col] == 7:
                lo -= 1
            while hi + 1 < H and grid[hi + 1, col] == 7:
                hi += 1
            if step == 1:
                end = lo
            else:
                end = hi
        else:
            # Farthest orange pixel in the inward direction.
            if step == 1:
                end = max(col_7s)
            else:
                end = min(col_7s)

        # Fill from marker to end (inclusive), preserving original 7s.
        if marker_row <= end:
            rng = range(marker_row, end + 1)
        else:
            rng = range(marker_row, end - 1, -1)

        for r in rng:
            if 0 <= r < H:
                if grid[r, col] != 7:
                    out[r, col] = 2
                else:
                    out[r, col] = 7

        # Ensure marker itself is red if it was not overwritten by a 7.
        if out[marker_row, col] != 7:
            out[marker_row, col] = 2

    def fill_horizontal_line(row, marker_col, side):
        if row is None or not (0 <= row < H):
            return
        if side == "left":
            step = 1
        elif side == "right":
            step = -1
        else:
            step = 1

        row_7s = [c for c in range(W) if grid[row, c] == 7]
        if not row_7s:
            if 0 <= marker_col < W:
                out[row, marker_col] = 2
            return

        if len(markers) == 1 and side in ("bottom", "right"):
            nearest = min(row_7s, key=lambda c: abs(c - marker_col))
            lo = nearest
            hi = nearest
            while lo - 1 >= 0 and grid[row, lo - 1] == 7:
                lo -= 1
            while hi + 1 < W and grid[row, hi + 1] == 7:
                hi += 1
            if step == 1:
                end = lo
            else:
                end = hi
        else:
            if step == 1:
                end = max(row_7s)
            else:
                end = min(row_7s)

        if marker_col <= end:
            rng = range(marker_col, end + 1)
        else:
            rng = range(marker_col, end - 1, -1)

        for c in rng:
            if 0 <= c < W:
                if grid[row, c] != 7:
                    out[row, c] = 2
                else:
                    out[row, c] = 7

        if out[row, marker_col] != 7:
            out[row, marker_col] = 2

    for r, c, side in markers:
        if side in ("top", "bottom"):
            fill_vertical_line(vc, r, side)
        elif side in ("left", "right"):
            fill_horizontal_line(hr, c, side)

    return out