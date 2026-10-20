import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Detect thick 1-regions and draw 7-borders around holes inside them.

    A pixel is considered "solid/supporting" if it belongs to a 2x2 block that is
    almost completely filled (at least 3 filled cells). Starting from the original
    1s, we iteratively mark a 0 as a hole when:
      - it can complete a 2x2 block,
      - it has at least 2 supporting filled neighbors,
      - counting grid border as support, it has at least 3 total supports.

    After all holes are found, keep only original 1s that lie in a full 2x2
    filled block in the final filled image. Then draw 7 on the 8-neighbors of
    active holes, leaving hole centers as 0.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return np.zeros_like(grid, dtype=int)

    h, w = grid.shape
    original_one = (grid == 1)
    zero_mask = (grid == 0)

    filled = original_one.copy()
    holes = np.zeros((h, w), dtype=bool)

    def compute_metrics(mask: np.ndarray):
        """
        Return:
          strict: cells in any 2x2 block with 4 filled cells
          loose:  cells in any 2x2 block with at least 3 filled cells
          fillable: 0-cells that are the missing cell of some 2x2 block with 3 filled cells
        """
        strict = np.zeros((h, w), dtype=bool)
        loose = np.zeros((h, w), dtype=bool)
        fillable = np.zeros((h, w), dtype=bool)

        if h >= 2 and w >= 2:
            count = (
                mask[:-1, :-1].astype(int) +
                mask[:-1, 1:].astype(int) +
                mask[1:, :-1].astype(int) +
                mask[1:, 1:].astype(int)
            )

            four = (count == 4)
            ok = (count >= 3)
            three = (count == 3)

            if np.any(four):
                strict[:-1, :-1] |= four
                strict[:-1, 1:] |= four
                strict[1:, :-1] |= four
                strict[1:, 1:] |= four

            if np.any(ok):
                loose[:-1, :-1] |= ok
                loose[:-1, 1:] |= ok
                loose[1:, :-1] |= ok
                loose[1:, 1:] |= ok

            if np.any(three):
                fillable[:-1, :-1] |= (three & ~mask[:-1, :-1])
                fillable[:-1, 1:]  |= (three & ~mask[:-1, 1:])
                fillable[1:, :-1]  |= (three & ~mask[1:, :-1])
                fillable[1:, 1:]   |= (three & ~mask[1:, 1:])

        return strict, loose, fillable

    # Iteratively fill holes that are supported by thick regions.
    max_iter = max(h, w) * 2 + 5
    for _ in range(max_iter):
        _, loose, fillable = compute_metrics(filled)

        # Only filled cells can provide support.
        support = loose & filled

        support_count = np.zeros((h, w), dtype=int)
        if h > 1:
            support_count[1:, :] += support[:-1, :]
            support_count[:-1, :] += support[1:, :]
        if w > 1:
            support_count[:, 1:] += support[:, :-1]
            support_count[:, :-1] += support[:, 1:]

        boundary = np.zeros((h, w), dtype=int)
        boundary[0, :] += 1
        boundary[-1, :] += 1
        boundary[:, 0] += 1
        boundary[:, -1] += 1

        total_support = support_count + boundary

        candidate_holes = (
            zero_mask &
            (~holes) &
            fillable &
            (support_count >= 2) &
            (total_support >= 3)
        )

        if not np.any(candidate_holes):
            break

        holes |= candidate_holes
        filled |= candidate_holes

    # Final object mask: original 1s that are in a full 2x2 block after filling.
    strict_final, _, _ = compute_metrics(filled)
    object_mask = original_one & strict_final

    # A hole is active if it is adjacent to at least one kept object pixel.
    active_holes = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(holes):
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        if np.any(object_mask[r0:r1, c0:c1]):
            active_holes[r, c] = True

    out = np.zeros((h, w), dtype=int)
    out[object_mask] = 1

    # Draw 3x3 borders of 7 around active holes, but never overwrite hole centers.
    for r, c in np.argwhere(active_holes):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not holes[nr, nc]:
                    out[nr, nc] = 7

    out[holes] = 0
    return out