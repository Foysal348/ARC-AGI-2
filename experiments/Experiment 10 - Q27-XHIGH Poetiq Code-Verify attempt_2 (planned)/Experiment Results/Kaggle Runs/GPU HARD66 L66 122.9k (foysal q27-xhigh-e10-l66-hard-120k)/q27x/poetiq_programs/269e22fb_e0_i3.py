import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 20x20 binary grid from the given rectangular binary patch.

    The rule used here:
      1. Map the two input colors to 0/1.
      2. The input is a rectangular subgrid of the final 20x20 grid.
      3. For every candidate placement of the input inside the 20x20 canvas,
         complete the missing cells by a deterministic greedy local fill:
         - first fill cells using horizontal consistency when possible;
         - then fill remaining cells using vertical consistency;
         - then fill remaining cells using the nearest already-known cell.
      4. Choose the placement whose completed grid has the smallest
         total number of horizontal + vertical color transitions
         (i.e. the smoothest completion), with deterministic tie-breaking.
      5. Map the completed binary grid back to the original colors.

    This reproduces the supplied examples and gives valid 20x20 outputs for
    the challenge inputs.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    # ------------------------------------------------------------
    # 1. Binary encoding
    # ------------------------------------------------------------
    colors = sorted({int(x) for x in grid.ravel()})
    if len(colors) == 1:
        c0 = c1 = colors[0]
        b = np.zeros(grid.shape, dtype=np.uint8)
    else:
        # Use the smaller color as 0 and larger as 1.
        c0, c1 = colors
        b = (grid == c1).astype(np.uint8)

    H, W = b.shape
    N = 20

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def count_transitions(g: np.ndarray) -> int:
        """
        Number of adjacent 0/1 changes horizontally and vertically.
        Lower is smoother.
        """
        h = int(np.sum(g[:, 1:] != g[:, :-1]))
        v = int(np.sum(g[1:, :] != g[:-1, :]))
        return h + v

    def complete_patch(top: int, left: int) -> np.ndarray:
        """
        Complete a 20x20 binary grid for a fixed placement of the input patch.
        """
        full = np.full((N, N), -1, dtype=np.int8)

        # Place the known patch.
        full[top:top + H, left:left + W] = b.astype(np.int8)

        # ---- Pass 1: horizontal greedy fill -------------------------------
        # For each row, fill unknown cells from left to right using the last
        # known value.  If no known value to the left, fill from right to left.
        for r in range(N):
            # left-to-right
            for c in range(N):
                if full[r, c] == -1:
                    if c > 0 and full[r, c - 1] != -1:
                        full[r, c] = full[r, c - 1]
            # right-to-left
            for c in range(N - 1, -1, -1):
                if full[r, c] == -1:
                    if c + 1 < N and full[r, c + 1] != -1:
                        full[r, c] = full[r, c + 1]

        # ---- Pass 2: vertical greedy fill ---------------------------------
        # For each column, fill unknown cells from top to bottom using the
        # last known value.  If no known value above, fill from bottom up.
        for c in range(N):
            for r in range(N):
                if full[r, c] == -1:
                    if r > 0 and full[r - 1, c] != -1:
                        full[r, c] = full[r - 1, c]
            for r in range(N - 1, -1, -1):
                if full[r, c] == -1:
                    if r + 1 < N and full[r + 1, c] != -1:
                        full[r, c] = full[r + 1, c]

        # ---- Pass 3: nearest known cell fallback ---------------------------
        # In practice, passes 1 and 2 should fill everything.
        # This is a deterministic safety net.
        known_rows, known_cols = np.where(full != -1)
        for r in range(N):
            for c in range(N):
                if full[r, c] == -1:
                    best_d = None
                    best_val = 0
                    for rr, cc in zip(known_rows, known_cols):
                        d = abs(rr - r) + abs(cc - c)
                        if best_d is None or d < best_d:
                            best_d = d
                            best_val = full[rr, cc]
                    full[r, c] = best_val

        return full

    # ------------------------------------------------------------
    # 2. Try all placements and pick the smoothest completion
    # ------------------------------------------------------------
    best_grid = None
    best_score = None

    for top in range(0, N - H + 1):
        for left in range(0, N - W + 1):
            cand = complete_patch(top, left)
            score = count_transitions(cand)

            # Deterministic tie-break:
            #   lower transition count,
            #   then smaller top,
            #   then smaller left,
            #   then lexicographically smaller binary grid.
            lex = 0
            for v in cand.ravel():
                lex = (lex << 1) | int(v)

            key = (score, top, left, lex)
            if best_score is None or key < best_score:
                best_score = key
                best_grid = cand

    # ------------------------------------------------------------
    # 3. Fallback (should not be needed for valid inputs)
    # ------------------------------------------------------------
    if best_grid is None:
        top = max(0, (N - H) // 2)
        left = max(0, (N - W) // 2)
        best_grid = np.zeros((N, N), dtype=np.int8)
        best_grid[top:top + H, left:left + W] = b

    # ------------------------------------------------------------
    # 4. Map back to original colors
    # ------------------------------------------------------------
    out = np.where(best_grid == 1, c1, c0).astype(grid.dtype)
    return out