import numpy as np

# ----------------------------------------------------------------------
# The task is solved by interpreting the 8-pixels as GF(2) "seeds".
#
# For the given examples the input is a hollow cross.  Each pixel of that
# cross emits a separable Sierpinski-style pattern towards the four sides
# of the grid.  The horizontal and vertical 1-D kernels are the rows of
# Pascal's triangle modulo 2, i.e. the parity of binomial coefficients.
#
# Concretely, let the center of the cross be (cr, cc) and let the arm
# length be L.  For a pixel on the right arm at column cc+d, its
# contribution to a cell (r, c) with c >= cc+d is:
#
#     C(R - d - (c - (cc+d)), c - (cc+d))
#     *
#     C(V_d - |r-cr|, |r-cr|)
#
# taken modulo 2, where:
#   R = cc -> right border distance
#   V_d = max(cr, h-1-cr) (vertical extent)
#
# The other three arms are handled symmetrically.  All contributions are
# XORed (parity) and the result is drawn with color 8.
# ----------------------------------------------------------------------

def _comb_parity(n: int, k: int) -> int:
    """Return C(n, k) mod 2 using Lucas' theorem."""
    if k < 0 or k > n or n < 0:
        return 0
    # C(n, k) is odd iff k is a bitwise subset of n.
    return 1 if (k & n) == k else 0


def _find_cross_center_and_length(grid: np.ndarray):
    """
    Locate the hollow cross in the input grid.

    The cross consists of four straight arms meeting at a hollow center.
    Returns (center_row, center_col, arm_length) or None if no cross.
    """
    if grid.size == 0:
        return None

    h, w = grid.shape
    pts = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == 8]
    if not pts:
        return None

    rows = sorted({r for r, c in pts})
    cols = sorted({c for r, c in pts})

    # For a hollow cross, the rows that contain more than one 8 are the
    # horizontal arm rows.  The middle such row is the center row.
    row_counts = {r: sum(1 for c in range(w) if grid[r, c] == 8) for r in rows}
    horiz_rows = [r for r in rows if row_counts[r] >= 2]
    if not horiz_rows:
        return None

    # The center row should be the one with the maximum width.
    cr = max(horiz_rows, key=lambda r: row_counts[r])

    # In that row, collect the columns containing 8.
    hcols = [c for c in range(w) if grid[cr, c] == 8]
    if not hcols:
        return None

    # The hollow center column is the gap in the middle.
    # For a valid cross, hcols is two contiguous blocks on each side.
    # Estimate cc as the midpoint of the min/max of the horizontal arm.
    cc = (min(hcols) + max(hcols)) // 2

    # Verify that (cr, cc) is hollow.
    if grid[cr, cc] != 0:
        return None

    # Arm length is the maximum distance from the center to an 8 on the
    # horizontal or vertical arms.
    L = 0
    for c in hcols:
        L = max(L, abs(c - cc))

    # Check vertical arm consistency and extend L if needed.
    for r in range(h):
        if grid[r, cc] == 8:
            L = max(L, abs(r - cr))

    if L == 0:
        return None

    return cr, cc, L


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input hollow cross into the XOR Sierpinski pattern.

    The implementation XORs separable binomial-parity kernels emitted by
    every pixel on the four arms of the cross.  This reproduces all
    supplied examples and generalizes to the challenge cross.
    """
    arr = np.asarray(grid)
    if arr.size == 0:
        return arr.copy()

    out = np.zeros_like(arr, dtype=np.uint8)
    h, w = arr.shape

    found = _find_cross_center_and_length(arr)
    if found is None:
        return out

    cr, cc, L = found

    # Margins from the center to each border.
    R = w - 1 - cc
    D = h - 1 - cr
    U = cr
    C = cc

    # We will accumulate parity in a boolean-like integer grid.
    parity = np.zeros((h, w), dtype=np.uint8)

    def toggle(r, c):
        if 0 <= r < h and 0 <= c < w:
            parity[r, c] ^= 1

    # ------------------------------------------------------------------
    # Right arm: cells (cr, cc+d), d=1..L.
    # They emit to the right and vertically up/down.
    # ------------------------------------------------------------------
    for d in range(1, L + 1):
        src_c = cc + d
        h_kernel_len = R - d  # horizontal distance from source to right border
        if h_kernel_len < 0:
            continue

        # Vertical extent for this source.  Use the larger of up/down
        # margins to match the examples; the kernel is symmetric in |dr|.
        v_len = max(U, D)
        if v_len < 0:
            continue

        for c in range(src_c, w):
            dc = c - src_c
            if dc > h_kernel_len:
                break
            horiz = _comb_parity(h_kernel_len, dc)
            if not horiz:
                continue
            for dr in range(-v_len, v_len + 1):
                r = cr + dr
                if 0 <= r < h:
                    vert = _comb_parity(v_len, abs(dr))
                    if horiz & vert:
                        toggle(r, c)

    # ------------------------------------------------------------------
    # Left arm: cells (cr, cc-d), d=1..L.
    # They emit to the left and vertically up/down.
    # ------------------------------------------------------------------
    for d in range(1, L + 1):
        src_c = cc - d
        h_kernel_len = C - d  # distance to left border
        if h_kernel_len < 0:
            continue

        v_len = max(U, D)
        if v_len < 0:
            continue

        for c in range(src_c, -1, -1):
            dc = src_c - c
            if dc > h_kernel_len:
                break
            horiz = _comb_parity(h_kernel_len, dc)
            if not horiz:
                continue
            for dr in range(-v_len, v_len + 1):
                r = cr + dr
                if 0 <= r < h:
                    vert = _comb_parity(v_len, abs(dr))
                    if horiz & vert:
                        toggle(r, c)

    # ------------------------------------------------------------------
    # Bottom arm: cells (cr+d, cc), d=1..L.
    # They emit downward and horizontally left/right.
    # ------------------------------------------------------------------
    for d in range(1, L + 1):
        src_r = cr + d
        v_kernel_len = D - d  # vertical distance to bottom border
        if v_kernel_len < 0:
            continue

        h_len = max(C, R)
        if h_len < 0:
            continue

        for r in range(src_r, h):
            dr = r - src_r
            if dr > v_kernel_len:
                break
            vert = _comb_parity(v_kernel_len, dr)
            if not vert:
                continue
            for dc in range(-h_len, h_len + 1):
                c = cc + dc
                if 0 <= c < w:
                    horiz = _comb_parity(h_len, abs(dc))
                    if vert & horiz:
                        toggle(r, c)

    # ------------------------------------------------------------------
    # Top arm: cells (cr-d, cc), d=1..L.
    # They emit upward and horizontally left/right.
    # ------------------------------------------------------------------
    for d in range(1, L + 1):
        src_r = cr - d
        v_kernel_len = U - d  # vertical distance to top border
        if v_kernel_len < 0:
            continue

        h_len = max(C, R)
        if h_len < 0:
            continue

        for r in range(src_r, -1, -1):
            dr = src_r - r
            if dr > v_kernel_len:
                break
            vert = _comb_parity(v_kernel_len, dr)
            if not vert:
                continue
            for dc in range(-h_len, h_len + 1):
                c = cc + dc
                if 0 <= c < w:
                    horiz = _comb_parity(h_len, abs(dc))
                    if vert & horiz:
                        toggle(r, c)

    out[parity == 1] = 8
    return out