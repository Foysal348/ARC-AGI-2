import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid as follows:

    1. Remove thin 1 pixels:
       Any 1 that is not part of at least one 2x2 all-1 block becomes 0.

    2. Identify target zero pixels:
       A 0 pixel in the original input is a target if it has at least three
       orthogonal (up/down/left/right) neighbors equal to 1.

    3. Paint the 8-neighborhood of every target zero with 7.
       Target zero pixels themselves remain 0.
       7 overrides both ordinary 0s and removed thin 1s.
    """
    g = np.array(grid, dtype=int)

    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    one = (g == 1)
    zero = (g == 0)

    # -------------------------------------------------------------------------
    # Step 1: keep only 1s that belong to at least one 2x2 all-1 block.
    # -------------------------------------------------------------------------
    keep_one = np.zeros((H, W), dtype=bool)

    if H >= 2 and W >= 2:
        block = (
            one[:-1, :-1] &
            one[:-1, 1:] &
            one[1:, :-1] &
            one[1:, 1:]
        )

        keep_one[:-1, :-1] |= block
        keep_one[:-1, 1:] |= block
        keep_one[1:, :-1] |= block
        keep_one[1:, 1:] |= block

    out = g.copy()
    out[one & ~keep_one] = 0

    # -------------------------------------------------------------------------
    # Step 2: find original 0 cells with at least 3 orthogonal 1 neighbors.
    # Out-of-bounds neighbors are treated as 0.
    # -------------------------------------------------------------------------
    orth_count = np.zeros((H, W), dtype=int)

    if H > 1:
        orth_count[1:, :] += one[:-1, :]
        orth_count[:-1, :] += one[1:, :]

    if W > 1:
        orth_count[:, 1:] += one[:, :-1]
        orth_count[:, :-1] += one[:, 1:]

    target_zero = zero & (orth_count >= 3)

    # -------------------------------------------------------------------------
    # Step 3: paint the 8-neighborhood of target zeros with 7.
    # Target zeros themselves must remain 0.
    # -------------------------------------------------------------------------
    for r, c in np.argwhere(target_zero):
        for dr in (-1, 0, 1):
            nr = r + dr
            if nr < 0 or nr >= H:
                continue

            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue

                nc = c + dc
                if 0 <= nc < W and not target_zero[nr, nc]:
                    out[nr, nc] = 7

    # Ensure target zero cells are exactly 0, even if they were touched
    # indirectly as neighbors of another target zero.
    out[target_zero] = 0

    return out