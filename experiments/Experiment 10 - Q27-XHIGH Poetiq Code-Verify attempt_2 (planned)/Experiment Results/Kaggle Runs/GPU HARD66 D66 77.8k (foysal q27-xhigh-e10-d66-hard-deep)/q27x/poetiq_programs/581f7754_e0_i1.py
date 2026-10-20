import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by aligning each movable object with its matching
    border reference pixel.

    Rule inferred from the examples:
      1. The background color is the most frequent color.
      2. For each non-background color c that appears on the border:
           - If the border pixels of c are only on the top/bottom borders,
             then every other pixel of c is moved horizontally into the
             reference column.
           - If the border pixels of c are only on the left/right borders,
             then every other pixel of c is moved vertically into the
             reference row.
           - If c appears on multiple border types, choose the axis that has
             the fewer border reference pixels; this happens only in
             contrived cases.
      3. The object associated with a movable pixel is the connected
         component of all non-background cells that contains that pixel.
         The whole component is translated so that its special pixel reaches
         the aligned position.
      4. Reference border pixels remain fixed.

    This implements the main behavior shown in the examples: objects
    containing a colored marker are slid until the marker lies in the same
    row/column as the matching marker on the border.
    """
    if grid.size == 0:
        return grid

    out = grid.astype(int, copy=True)
    H, W = grid.shape

    # Background = most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Determine reference information for every non-background color.
    # For each color, record:
    #   - whether it has border references
    #   - axis: 'h' means align columns (references top/bottom)
    #   - target coordinate: column for 'h', row for 'v'
    refs = {}

    colors = set(int(c) for c in np.unique(grid) if c != bg)
    for c in colors:
        mask = (grid == c)
        if not np.any(mask):
            continue

        top = np.any(mask[0, :])
        bottom = np.any(mask[-1, :])
        left = np.any(mask[:, 0])
        right = np.any(mask[:, -1])

        tb = int(top) + int(bottom)
        lr = int(left) + int(right)

        if tb == 0 and lr == 0:
            continue

        if tb > 0 and lr == 0:
            axis = 'h'
            # Use the column of the first border pixel found.
            # If several columns exist, use their average rounded to nearest int.
            cols = np.where(mask[0, :] | mask[-1, :])[0]
            target = int(round(float(np.mean(cols))))
        elif lr > 0 and tb == 0:
            axis = 'v'
            rows = np.where(mask[:, 0] | mask[:, -1])[0]
            target = int(round(float(np.mean(rows))))
        elif tb == 0 and lr == 0:
            continue
        else:
            # Mixed border references: pick the side with fewer reference cells.
            if tb <= lr:
                axis = 'h'
                cols = np.where(mask[0, :] | mask[-1, :])[0]
                target = int(round(float(np.mean(cols))))
            else:
                axis = 'v'
                rows = np.where(mask[:, 0] | mask[:, -1])[0]
                target = int(round(float(np.mean(rows))))

        refs[c] = (axis, target)

    # Connected component labeling for non-background cells (4-connectivity).
    comp_id = np.full((H, W), -1, dtype=int)
    components = []

    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg or comp_id[r, c] != -1:
                continue

            cid = len(components)
            q = deque([(r, c)])
            comp_id[r, c] = cid
            cells = [(r, c)]

            while q:
                cr, cc = q.popleft()
                for nr, nc in ((cr + 1, cc), (cr - 1, cc),
                               (cr, cc + 1), (cr, cc - 1)):
                    if 0 <= nr < H and 0 <= nc < W:
                        if comp_id[nr, nc] == -1 and grid[nr, nc] != bg:
                            comp_id[nr, nc] = cid
                            q.append((nr, nc))
                            cells.append((nr, nc))

            components.append(cells)

    # For each component, see whether it contains a movable reference color.
    # If so, compute the displacement needed to align that special pixel.
    # If multiple special pixels exist, use the one whose required displacement
    # is smallest; this is conservative and matches the examples where each
    # component has exactly one relevant marker.
    for cid, cells in enumerate(components):
        best_move = None
        best_color = None
        best_pos = None

        for (r, c) in cells:
            col = int(grid[r, c])
            if col == bg or col not in refs:
                continue

            # Do not move border reference pixels.
            is_border = (r == 0 or r == H - 1 or c == 0 or c == W - 1)
            if is_border:
                continue

            axis, target = refs[col]
            if axis == 'h':
                dr, dc = 0, target - c
            else:
                dr, dc = target - r, 0

            # Prefer smaller absolute displacement; ties arbitrary.
            score = abs(dr) + abs(dc)
            if best_move is None or score < best_move[2]:
                best_move = (dr, dc, score)
                best_color = col
                best_pos = (r, c)

        if best_move is None:
            continue

        dr, dc, _ = best_move

        # Apply the translation to the whole component, if it stays in bounds
        # and does not create an obvious collision with an unmodified cell.
        new_cells = []
        valid = True
        for (r, c) in cells:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                valid = False
                break
            new_cells.append((nr, nc))

        if not valid:
            continue

        # If a moved pixel would overwrite a different, non-background pixel
        # that is not part of this component, skip moving this component.
        target_ids = set(comp_id[r, c] for (r, c) in cells)
        for (nr, nc) in new_cells:
            if comp_id[nr, nc] != -1 and comp_id[nr, nc] not in target_ids:
                valid = False
                break

        if not valid:
            continue

        # Copy component cells to their new positions.
        for (r, c), (nr, nc) in zip(cells, new_cells):
            out[nr, nc] = grid[r, c]

        # Clear original positions that are no longer occupied by this component.
        new_set = set(new_cells)
        for (r, c) in cells:
            if (r, c) not in new_set:
                out[r, c] = bg

    return out