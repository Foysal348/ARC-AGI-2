import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid as follows:

    1. A 1-pixel is "thick" if it belongs to at least one 2x2 all-1 block.
       Thick 1s form the stable structure of the image.

    2. Clean background noise:
       Repeatedly remove 4-connected 1-components that:
         - contain no thick 1-pixel, and
         - touch the grid border or are orthogonally adjacent to a zero
           component that touches the grid border.
       This removes isolated/thin 1-noise in the outside background,
       while preserving 1-structures that have a thick part.

    3. A zero-pixel is "thick" if it belongs to at least one 2x2 all-zero
       block after the background cleaning step.

    4. A zero-pixel is a "crack/hole center" if:
         - it was originally zero,
         - it is not a thick zero, and
         - it has at least two orthogonal neighbors that are thick 1s.

    5. Paint the 8-neighborhood of every center with color 7.
       The centers themselves remain 0.

    The final grid contains 0, 1, and 7.
    """
    g = np.array(grid, dtype=int)
    if g.ndim != 2:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g.copy()

    one = (g == 1)
    original_zero = (g == 0)

    # ------------------------------------------------------------------
    # Thick 1s: 1s that are in at least one 2x2 all-1 square.
    # ------------------------------------------------------------------
    thick_one = np.zeros((H, W), dtype=bool)
    if H >= 2 and W >= 2:
        block = (
            one[:-1, :-1] &
            one[:-1, 1:] &
            one[1:, :-1] &
            one[1:, 1:]
        )
        thick_one[:-1, :-1] |= block
        thick_one[:-1, 1:] |= block
        thick_one[1:, :-1] |= block
        thick_one[1:, 1:] |= block

    clean = g.copy()

    # ------------------------------------------------------------------
    # Helper: return mask of zeros connected to the grid border (4-way).
    # ------------------------------------------------------------------
    def background_zero_mask(img: np.ndarray) -> np.ndarray:
        zero = (img == 0)
        bg = np.zeros((H, W), dtype=bool)
        q = deque()

        # Seed with all border zeros.
        for r in (0, H - 1):
            for c in range(W):
                if 0 <= r < H and zero[r, c] and not bg[r, c]:
                    bg[r, c] = True
                    q.append((r, c))

        for c in (0, W - 1):
            for r in range(H):
                if 0 <= c < W and zero[r, c] and not bg[r, c]:
                    bg[r, c] = True
                    q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and zero[nr, nc] and not bg[nr, nc]:
                    bg[nr, nc] = True
                    q.append((nr, nc))

        return bg

    # ------------------------------------------------------------------
    # Remove background thin 1-components.
    #
    # A 1-component is removable if it has no thick 1-cell and is exposed
    # to the outside background: it either touches the grid border or has
    # a 4-neighbor zero in a border-connected zero component.
    # ------------------------------------------------------------------
    for _ in range(H * W + 1):
        bg_zero = background_zero_mask(clean)

        labels = np.zeros((H, W), dtype=int)
        removable = np.zeros((H, W), dtype=bool)
        comp_id = 1

        for r in range(H):
            for c in range(W):
                if clean[r, c] != 1 or labels[r, c] != 0:
                    continue

                q = deque([(r, c)])
                labels[r, c] = comp_id
                cells = []
                has_thick = False
                touches_border = False
                adjacent_to_bg_zero = False

                while q:
                    cr, cc = q.popleft()
                    cells.append((cr, cc))

                    if cr == 0 or cr == H - 1 or cc == 0 or cc == W - 1:
                        touches_border = True
                    if thick_one[cr, cc]:
                        has_thick = True

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if not (0 <= nr < H and 0 <= nc < W):
                            continue

                        if clean[nr, nc] == 1 and labels[nr, nc] == 0:
                            labels[nr, nc] = comp_id
                            q.append((nr, nc))
                        elif clean[nr, nc] == 0 and bg_zero[nr, nc]:
                            adjacent_to_bg_zero = True

                if (not has_thick) and (touches_border or adjacent_to_bg_zero):
                    for cr, cc in cells:
                        removable[cr, cc] = True

                comp_id += 1

        if not removable.any():
            break

        clean[removable] = 0

    # ------------------------------------------------------------------
    # Thick zeros: zeros that are in at least one 2x2 all-zero square
    # in the cleaned grid.
    # ------------------------------------------------------------------
    thick_zero = np.zeros((H, W), dtype=bool)
    zero_clean = (clean == 0)
    if H >= 2 and W >= 2:
        block0 = (
            zero_clean[:-1, :-1] &
            zero_clean[:-1, 1:] &
            zero_clean[1:, :-1] &
            zero_clean[1:, 1:]
        )
        thick_zero[:-1, :-1] |= block0
        thick_zero[:-1, 1:] |= block0
        thick_zero[1:, :-1] |= block0
        thick_zero[1:, 1:] |= block0

    # ------------------------------------------------------------------
    # Count orthogonal thick-1 neighbors for every cell.
    # ------------------------------------------------------------------
    orth_thick_one_count = np.zeros((H, W), dtype=int)
    if H > 1:
        orth_thick_one_count[1:, :] += thick_one[:-1, :]
        orth_thick_one_count[:-1, :] += thick_one[1:, :]
    if W > 1:
        orth_thick_one_count[:, 1:] += thick_one[:, :-1]
        orth_thick_one_count[:, :-1] += thick_one[:, 1:]

    # ------------------------------------------------------------------
    # Centers: original thin zero pixels pinched by at least two thick 1s.
    # ------------------------------------------------------------------
    centers = original_zero & (~thick_zero) & (orth_thick_one_count >= 2)

    out = clean.copy()

    # ------------------------------------------------------------------
    # Paint 7 around all centers, but leave centers as 0.
    # ------------------------------------------------------------------
    for r, c in np.argwhere(centers):
        for dr in (-1, 0, 1):
            nr = r + dr
            if nr < 0 or nr >= H:
                continue
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nc = c + dc
                if 0 <= nc < W and not centers[nr, nc]:
                    out[nr, nc] = 7

    # Ensure centers are exactly 0, even if touched by neighboring centers.
    out[centers] = 0

    return out