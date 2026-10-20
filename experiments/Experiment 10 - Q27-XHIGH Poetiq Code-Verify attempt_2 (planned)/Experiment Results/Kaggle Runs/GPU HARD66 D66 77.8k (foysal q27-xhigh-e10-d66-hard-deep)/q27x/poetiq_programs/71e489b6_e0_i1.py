import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid as follows:

    1. Clean the grid by removing single-pixel 1 "noise" pixels that are
       part of the background zero region (i.e. the zero region reachable
       from them touches the grid border).

    2. Identify "target" 0 pixels.
       A 0 pixel is a target if:
       - it lies on a thin zero part, meaning it is not part of any 2x2
         all-zero block, and it belongs to a zero run that is bounded by
         1s/border in one axis and has perpendicular support from a 1,
         OR
       - it belongs to a very small zero component (area <= 4) that is
         similarly bounded/supported.

       This marks single-pixel notches, 1-pixel cracks, small holes,
       and small 2x2 holes, while ignoring large zero regions.

    3. Paint the 8-neighbourhood of every target 0 pixel with 7.
       Target 0 pixels themselves remain 0.

    The final output contains only 0, 1, and 7.
    """

    grid = np.array(grid, dtype=int)
    if grid.size == 0:
        return grid

    H, W = grid.shape
    original_zero = (grid == 0)

    # -------------------------------------------------------------------------
    # Helper: 4-connected component labels for a binary mask.
    # Returns labels array and list of component coordinates.
    # -------------------------------------------------------------------------
    def component_labels(mask):
        labels = np.zeros((H, W), dtype=int)
        comps = []
        idx = 1

        for r in range(H):
            for c in range(W):
                if mask[r, c] and labels[r, c] == 0:
                    q = deque([(r, c)])
                    labels[r, c] = idx
                    cells = []
                    while q:
                        cr, cc = q.popleft()
                        cells.append((cr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W:
                                if mask[nr, nc] and labels[nr, nc] == 0:
                                    labels[nr, nc] = idx
                                    q.append((nr, nc))
                    comps.append(cells)
                    idx += 1

        return labels, comps

    # -------------------------------------------------------------------------
    # Step 1: iterative removal of background single-pixel 1s.
    #
    # A single 1 is considered background noise if:
    # - it is on the border, or
    # - one of its 4-neighbours is a zero in a zero component that touches
    #   the border.
    #
    # We iterate because removing one pixel can make another isolated pixel
    # connected to background.
    # -------------------------------------------------------------------------
    cleaned = grid.copy()

    def remove_background_singles_once(img):
        # Zero components and whether they touch the border.
        zero_mask = (img == 0)
        zero_labels, zero_comps = component_labels(zero_mask)
        zero_touch = [False] * (len(zero_comps) + 1)
        for i, cells in enumerate(zero_comps, start=1):
            for r, c in cells:
                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    zero_touch[i] = True
                    break

        # Single-pixel 1 components.
        one_mask = (img == 1)
        _, one_comps = component_labels(one_mask)

        changed = False
        for cells in one_comps:
            if len(cells) != 1:
                continue

            r, c = cells[0]

            # Border single 1 is background noise.
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                img[r, c] = 0
                changed = True
                continue

            # If any 4-adjacent zero belongs to border-touching zero region,
            # this single 1 is noise.
            is_background = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if img[nr, nc] == 0 and zero_touch[zero_labels[nr, nc]]:
                        is_background = True
                        break

            if is_background:
                img[r, c] = 0
                changed = True

        return changed

    # Iterate to a fixed point, but guard against pathological infinite loops.
    max_iter = H * W + 5
    for _ in range(max_iter):
        if not remove_background_singles_once(cleaned):
            break

    # -------------------------------------------------------------------------
    # Step 2: identify target zero pixels.
    # -------------------------------------------------------------------------
    zero_mask = (cleaned == 0)

    # 2a. Zero component areas.
    zero_labels, zero_comps = component_labels(zero_mask)
    zero_area = np.zeros((H, W), dtype=int)
    for i, cells in enumerate(zero_comps, start=1):
        for r, c in cells:
            zero_area[r, c] = len(cells)

    # 2b. Cells that are part of at least one 2x2 all-zero block.
    in_2x2_zero = np.zeros((H, W), dtype=bool)
    if H >= 2 and W >= 2:
        block = (
            zero_mask[:-1, :-1] &
            zero_mask[:-1, 1:] &
            zero_mask[1:, :-1] &
            zero_mask[1:, 1:]
        )
        in_2x2_zero[:-1, :-1] |= block
        in_2x2_zero[:-1, 1:] |= block
        in_2x2_zero[1:, :-1] |= block
        in_2x2_zero[1:, 1:] |= block

    # 2c. Horizontal zero runs bounded by 1s or borders.
    h_flanked = np.zeros((H, W), dtype=bool)
    for r in range(H):
        c = 0
        while c < W:
            if cleaned[r, c] == 0:
                L = c
                while c < W and cleaned[r, c] == 0:
                    c += 1
                R = c - 1
                left_ok = (L == 0) or (cleaned[r, L - 1] == 1)
                right_ok = (R == W - 1) or (cleaned[r, R + 1] == 1)
                if left_ok and right_ok:
                    h_flanked[r, L:R + 1] = True
            else:
                c += 1

    # 2d. Vertical zero runs bounded by 1s or borders.
    v_flanked = np.zeros((H, W), dtype=bool)
    for c in range(W):
        r = 0
        while r < H:
            if cleaned[r, c] == 0:
                T = r
                while r < H and cleaned[r, c] == 0:
                    r += 1
                B = r - 1
                top_ok = (T == 0) or (cleaned[T - 1, c] == 1)
                bottom_ok = (B == H - 1) or (cleaned[B + 1, c] == 1)
                if top_ok and bottom_ok:
                    v_flanked[T:B + 1, c] = True
            else:
                r += 1

    # 2e. Perpendicular support: at least one 1 immediately above/below,
    #     and at least one 1 immediately left/right.
    v_support = np.zeros((H, W), dtype=bool)
    if H >= 2:
        v_support[1:, :] |= (cleaned[:-1, :] == 1)
        v_support[:-1, :] |= (cleaned[1:, :] == 1)

    h_support = np.zeros((H, W), dtype=bool)
    if W >= 2:
        h_support[:, 1:] |= (cleaned[:, :-1] == 1)
        h_support[:, :-1] |= (cleaned[:, 1:] == 1)

    # A zero is a target if it is on a supported zero run, and either the
    # zero component is tiny or the cell is not in a thick 2x2 zero area.
    supported_run = (
        (h_flanked & v_support) |
        (v_flanked & h_support)
    )
    small_or_thin = (zero_area <= 4) | (~in_2x2_zero)

    # Only original 0 cells can be target hole centers.
    target_zero = original_zero & zero_mask & supported_run & small_or_thin

    # -------------------------------------------------------------------------
    # Step 3: paint 7 around all target zero cells.
    # -------------------------------------------------------------------------
    out = cleaned.copy()

    for r in range(H):
        for c in range(W):
            if not target_zero[r, c]:
                continue

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        # Do not overwrite other target zero pixels.
                        if not target_zero[nr, nc]:
                            out[nr, nc] = 7

    return out