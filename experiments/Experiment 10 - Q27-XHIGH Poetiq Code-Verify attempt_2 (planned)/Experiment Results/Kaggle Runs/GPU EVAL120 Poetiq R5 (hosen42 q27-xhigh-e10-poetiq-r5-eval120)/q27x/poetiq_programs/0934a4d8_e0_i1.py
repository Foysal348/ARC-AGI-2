import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recover the rectangle hidden by 8s.

    The input contains one rectangular patch filled with 8s.  The rest of the
    grid contains the original values repeated by a symmetry such as:
      - vertical reflection,
      - horizontal reflection,
      - 180-degree rotation,
      - transpose,
      - anti-transpose.

    This function finds the bounding box of the 8s, tries several candidate
    symmetry mappings, and scores each mapping by checking whether the cells
    just outside the hidden box match the corresponding cells just outside the
    mapped source box.  The mapping with the best border consistency is used
    to fill the hidden patch, and only that patch is returned.
    """
    grid = np.array(grid, dtype=int)
    H, W = grid.shape

    if grid.size == 0:
        return grid.copy()

    mask = (grid == 8)
    if not mask.any():
        return grid.copy()

    # Bounding box of the 8-patch.
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h, w = r1 - r0 + 1, c1 - c0 + 1

    # Candidate transformations.
    candidates = []

    # Vertical reflections.  The examples use the offset reflection
    # r -> H + 1 - r, but include the standard one as a fallback.
    candidates.append(("vert_plus", lambda r, c: (H + 1 - r, c)))
    candidates.append(("vert_std", lambda r, c: (H - 1 - r, c)))

    # Horizontal reflections.
    candidates.append(("horiz_plus", lambda r, c: (r, W + 1 - c)))
    candidates.append(("horiz_std", lambda r, c: (r, W - 1 - c)))

    # 180-degree rotations.
    candidates.append(("rot_plus", lambda r, c: (H + 1 - r, W + 1 - c)))
    candidates.append(("rot_std", lambda r, c: (H - 1 - r, W - 1 - c)))

    # Transpose and anti-transpose, valid for square grids.
    if H == W:
        candidates.append(("transpose", lambda r, c: (c, r)))
        candidates.append(("anti_transpose_std", lambda r, c: (W - 1 - c, H - 1 - r)))
        candidates.append(("anti_transpose_plus", lambda r, c: (W + 1 - c, H + 1 - r)))

    def evaluate(mapping):
        """
        Return (output_patch, score) for a mapping, or (None, -inf) if the
        mapped source patch is invalid.
        """
        out = np.empty((h, w), dtype=int)

        # Map the whole hidden box to its candidate source location.
        source_cells = []
        sr_min = sr_max = sc_min = sc_max = None

        for i in range(h):
            for j in range(w):
                r, c = r0 + i, c0 + j
                sr, sc = mapping(r, c)

                if not (0 <= sr < H and 0 <= sc < W):
                    return None, float("-inf")

                # Source cannot itself be masked.
                if grid[sr, sc] == 8:
                    return None, float("-inf")

                out[i, j] = grid[sr, sc]
                source_cells.append((sr, sc))

                if sr_min is None:
                    sr_min = sr_max = sr
                    sc_min = sc_max = sc
                else:
                    sr_min = min(sr_min, sr)
                    sr_max = max(sr_max, sr)
                    sc_min = min(sc_min, sc)
                    sc_max = max(sc_max, sc)

        # Score by comparing a one-cell frame outside the hidden box with the
        # corresponding frame outside the source box.
        matches = 0
        mismatches = 0

        for r in range(max(0, r0 - 1), min(H, r1 + 2)):
            for c in range(max(0, c0 - 1), min(W, c1 + 2)):
                # Only consider cells just outside the hidden rectangle.
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue

                if grid[r, c] == 8:
                    continue

                sr, sc = mapping(r, c)

                if not (0 <= sr < H and 0 <= sc < W):
                    continue
                if grid[sr, sc] == 8:
                    continue

                # Ignore cells mapping into the interior of the source patch.
                if sr_min <= sr <= sr_max and sc_min <= sc <= sc_max:
                    continue

                if grid[r, c] == grid[sr, sc]:
                    matches += 1
                else:
                    mismatches += 1

        return out, matches - mismatches

    best_patch = None
    best_score = float("-inf")

    for _, mapping in candidates:
        patch, score = evaluate(mapping)
        if patch is not None and score > best_score:
            best_score = score
            best_patch = patch

    # Fallback: if no symmetry was convincingly found, return zeros for the
    # hidden box.  Valid task instances should not reach this branch.
    if best_patch is None:
        best_patch = np.zeros((h, w), dtype=int)

    return best_patch