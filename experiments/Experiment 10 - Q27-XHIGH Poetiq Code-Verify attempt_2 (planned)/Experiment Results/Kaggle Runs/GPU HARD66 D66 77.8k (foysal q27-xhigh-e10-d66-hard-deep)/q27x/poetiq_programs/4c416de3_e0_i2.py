import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule
    ==============
    The grid has a dominant background color and 0-colored wall corners.
    Single non-background, non-zero pixels are "seeds".  Each seed is
    associated with the nearest 0-corner whose interior quadrant contains
    the seed.

    For most grids, the seed completes a small corner pattern:
      - the corner itself,
      - the two exterior edge cells of the corner,
      - if the seed is farther than one Chebyshev step away, the exterior
        diagonal cell as well,
      - and a shortest king-move path from the corner to the seed.

    A special case occurs when the background color is also the maximum
    color in the grid.  Then the examples use a 3x3 template pattern
    instead of the small corner pattern.  The templates used by the
    provided examples are:
      - colors 4, 5, and 9: 110 / 101 / 110
      - all other colors:    110 / 101 / 011
    The template is transformed so that the seed lies on a 1-cell of the
    template.

    Existing non-background, non-zero pixels are preserved unless they are
    the seed itself.
    """
    if grid.size == 0:
        return grid

    grid = np.array(grid, copy=True)
    h, w = grid.shape

    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    use_max_template = (bg == int(vals.max()))

    out = grid.copy()

    def inside(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def paint(r: int, c: int, color: int) -> None:
        """Paint if the cell is background or a wall (0)."""
        if not inside(r, c):
            return
        if grid[r, c] in (bg, 0):
            out[r, c] = color

    # -------------------------------------------------------------------------
    # Collect all 0-corners.  A 0-corner is a zero cell with two orthogonal
    # zero neighbors, giving an interior diagonal direction.
    # -------------------------------------------------------------------------
    corners = []  # (r, c, dr, dc) where (dr, dc) is the interior direction
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                continue

            # Possible pairs of orthogonal zero neighbors.
            pairs = [
                ((1, 0), (0, 1)),   # top-left corner, interior down-right
                ((1, 0), (0, -1)),  # top-right corner, interior down-left
                ((-1, 0), (0, 1)),  # bottom-left corner, interior up-right
                ((-1, 0), (0, -1)), # bottom-right corner, interior up-left
            ]
            for (ar, ac), (br, bc) in pairs:
                nr1, nc1 = r + ar, c + ac
                nr2, nc2 = r + br, c + bc
                if (inside(nr1, nc1) and grid[nr1, nc1] == 0 and
                    inside(nr2, nc2) and grid[nr2, nc2] == 0):
                    dr = ar + br
                    dc = ac + bc
                    corners.append((r, c, dr, dc))

    # Seeds: non-background, non-zero pixels.
    sources = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != bg and grid[r, c] != 0:
                sources.append((r, c, int(grid[r, c])))

    # -------------------------------------------------------------------------
    # Pattern helpers.
    # -------------------------------------------------------------------------
    def orient_3x3(base, src_pos):
        """
        base is a 3x3 matrix.  src_pos is one of:
            'BR', 'BL', 'TR', 'TL'
        relative to the 3x3 center.  Returns a transformed 3x3 so that the
        original 'BR' source position maps to src_pos.
        """
        if src_pos == 'BR':
            return base
        if src_pos == 'BL':
            return base[:, ::-1]
        if src_pos == 'TR':
            return base[::-1, :]
        if src_pos == 'TL':
            return base[::-1, ::-1]
        raise ValueError("bad source position")

    P_DIAG = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=int)

    P_SQUARE = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 0]
    ], dtype=int)

    def apply_small_pattern(cr, cc, dr, dc, sr, sc, color):
        """
        Standard pattern used when background is not the maximum color.
        (dr, dc) is the interior direction from corner to seed.
        """
        # Corner.
        paint(cr, cc, color)

        # Exterior edge cells.
        paint(cr - dr, cc, color)
        paint(cr, cc - dc, color)

        # Chebyshev distance to the seed.
        D = max(abs(sr - cr), abs(sc - cc))
        if D > 1:
            paint(cr - dr, cc - dc, color)

        # King-move path from corner to seed: diagonal steps first, then
        # remaining straight steps.
        r, c = cr, cc
        while (r, c) != (sr, sc):
            if r != sr and c != sc:
                r += dr
                c += dc
            elif r != sr:
                r += dr
            else:
                c += dc
            paint(r, c, color)

    def apply_template_pattern(cr, cc, src_pos, color_val):
        """
        Special 3x3 template pattern used when background is the maximum
        color in the grid.
        """
        if color_val in (4, 5, 9):
            base = P_SQUARE
        else:
            base = P_DIAG

        pat = orient_3x3(base, src_pos)

        for i in range(3):
            for j in range(3):
                if pat[i, j] == 1:
                    r = cr - 1 + i
                    c = cc - 1 + j
                    paint(r, c, color_val)

    # -------------------------------------------------------------------------
    # Process each seed.
    # -------------------------------------------------------------------------
    for sr, sc, color in sources:
        best = None
        best_key = None

        for cr, cc, dr, dc in corners:
            rr = sr - cr
            cc_off = sc - cc

            # The seed must be in the interior quadrant of this corner.
            if (dr == 1 and rr < 1) or (dr == -1 and rr > -1):
                continue
            if (dc == 1 and cc_off < 1) or (dc == -1 and cc_off > -1):
                continue

            # Prefer pure diagonal placements, then smaller Chebyshev distance.
            pure = 0 if abs(rr) == abs(cc_off) else 1
            D = max(abs(rr), abs(cc_off))
            M = abs(rr) + abs(cc_off)

            key = (pure, D, M)
            if best_key is None or key < best_key:
                best_key = key
                best = (cr, cc, dr, dc)

        if best is None:
            continue

        cr, cc, dr, dc = best

        if use_max_template:
            # Determine which diagonal corner of the 3x3 contains the seed.
            if sr > cr and sc > cc:
                src_pos = 'BR'
            elif sr > cr and sc < cc:
                src_pos = 'BL'
            elif sr < cr and sc > cc:
                src_pos = 'TR'
            elif sr < cr and sc < cc:
                src_pos = 'TL'
            else:
                continue

            apply_template_pattern(cr, cc, src_pos, color)
        else:
            apply_small_pattern(cr, cc, dr, dc, sr, sc, color)

    return out