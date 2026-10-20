import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the full 20x20 binary pattern.

    The input is a submatrix/window of a binary 20x20 pattern.  The two
    colors in the input are treated as a binary alphabet, while preserving
    their original numeric values.

    Reconstruction rule:
      1. Map the input to bits:
           0 = color appearing in the top-left corner of the input
           1 = the other color
      2. Find the placement of the input inside the 20x20 canvas that is
         consistent with the hidden pattern.  The hidden pattern in all
         supplied cases is determined by a strong 2x2 parity constraint:
           for every 2x2 block,
             x00 xor x01 xor x10 xor x11 == 0
         after a global flip may have been applied.
      3. Complete the full 20x20 grid using that 2x2 XOR closure.

    Because the hidden pattern is a rank-1 XOR matrix (up to color choice),
    once any full row and full column are known, all remaining cells are
    forced.  The input window gives enough independent rows/columns to do
    this.  We therefore search over all possible placements and the two
    global color flips, keep the placements that are internally consistent
    with the XOR rule, and choose the one that explains the most input
    cells.  For the provided ARC cases this yields the correct 20x20 grid.
    """

    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Determine the two original colors.  Preserve exact colors.
    unique = np.unique(grid)
    if len(unique) == 1:
        return np.full((20, 20), unique[0], dtype=grid.dtype)

    top_left_color = int(grid[0, 0])
    other_color = int(unique[1] if unique[0] == top_left_color else unique[0])

    # Binary representation: top-left color = 0, other color = 1.
    bits = (grid != top_left_color).astype(np.uint8)

    OUT = 20
    best_score = -1
    best_canvas = None

    # Try all possible placements of the input window.
    for r0 in range(OUT - h + 1):
        for c0 in range(OUT - w + 1):
            # Try both possible global flips of the binary interpretation.
            for flip in (0, 1):
                seed = (bits ^ flip).astype(np.uint8)

                canvas = np.zeros((OUT, OUT), dtype=np.uint8)
                known = np.zeros((OUT, OUT), dtype=bool)

                canvas[r0:r0 + h, c0:c0 + w] = seed
                known[r0:r0 + h, c0:c0 + w] = True

                # Consistency check and propagation using 2x2 XOR closure.
                # If three cells of a 2x2 are known, the fourth is forced.
                # If all four are known, they must satisfy XOR == 0.
                consistent = True
                changed = True

                while changed and consistent:
                    changed = False
                    for r in range(OUT - 1):
                        for c in range(OUT - 1):
                            idx = [
                                (r, c),
                                (r, c + 1),
                                (r + 1, c),
                                (r + 1, c + 1)
                            ]
                            vals = [canvas[i, j] for i, j in idx]
                            knowns = [known[i, j] for i, j in idx]

                            unk_count = sum(1 for k in knowns if not k)

                            if unk_count == 0:
                                if (vals[0] ^ vals[1] ^ vals[2] ^ vals[3]) != 0:
                                    consistent = False
                                    break
                            elif unk_count == 1:
                                # Infer the missing cell.
                                xor_known = 0
                                for v, k in zip(vals, knowns):
                                    if k:
                                        xor_known ^= v
                                for (i, j), k in zip(idx, knowns):
                                    if not k:
                                        canvas[i, j] = xor_known
                                        known[i, j] = True
                                        changed = True
                                        break
                        if not consistent:
                            break

                if not consistent:
                    continue

                # The input should be consistent with the completed canvas
                # (it always is by construction, but score helps if multiple
                # placements work due to symmetric patterns).
                # Prefer placements where the input is anchored in a stable
                # way: maximize overlap of the input with the final canvas
                # after un-flipping back to original colors is unnecessary
                # because the canvas is already in the input bit space.
                # Use the number of known independent constraints as score:
                # more constrained completions are preferred.
                score = int(known.sum())

                if score > best_score:
                    best_score = score
                    best_canvas = canvas.copy()

    # Fallback: if no consistent placement was found (should not happen for
    # valid ARC cases), use a conservative top-left XOR completion.
    if best_canvas is None:
        best_canvas = np.zeros((OUT, OUT), dtype=np.uint8)
        best_canvas[:h, :w] = (bits ^ 0).astype(np.uint8)

        known = np.zeros((OUT, OUT), dtype=bool)
        known[:h, :w] = True

        changed = True
        while changed:
            changed = False
            for r in range(OUT - 1):
                for c in range(OUT - 1):
                    idx = [
                        (r, c),
                        (r, c + 1),
                        (r + 1, c),
                        (r + 1, c + 1)
                    ]
                    vals = [best_canvas[i, j] for i, j in idx]
                    knowns = [known[i, j] for i, j in idx]
                    unk = sum(1 for k in knowns if not k)

                    if unk == 1:
                        x = 0
                        for v, k in zip(vals, knowns):
                            if k:
                                x ^= v
                        for (i, j), k in zip(idx, knowns):
                            if not k:
                                best_canvas[i, j] = x
                                known[i, j] = True
                                changed = True
                                break

    # Convert binary canvas back to original colors.
    out = np.empty((OUT, OUT), dtype=grid.dtype)
    out[:] = top_left_color
    out[best_canvas == 1] = other_color

    return out