import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the one-pixel guide line hidden in the large colored region.

    The examples follow this pattern:
      * one background color,
      * one large "thick" region color,
      * a small row of 2x2 blocks that gives the repeating color sequence,
      * one other 2x2 block whose color is used as the endpoint / turn marker,
      * one or more single anchor pixels of the first sequence color on the left border.

    The implementation:
      1. detects background / main region / sequence / marker / anchors,
      2. extracts a 1-pixel skeleton from the thick main region,
      3. attaches the skeleton to the anchor pixels,
      4. colors the skeleton by walking from the anchors with the repeating sequence,
      5. places the marker color just outside the main region at path ends.

    The code is intentionally defensive: if the expected key objects are missing,
    it returns a copy of the input rather than crashing.
    """

    def components(mask: np.ndarray):
        """Return a list of 4-connected components as coordinate lists."""
        H, W = mask.shape
        label = np.full((H, W), -1, dtype=int)
        comps = []
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for r in range(H):
            for c in range(W):
                if not mask[r, c] or label[r, c] != -1:
                    continue
                cid = len(comps)
                q = deque([(r, c)])
                label[r, c] = cid
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and label[nx, ny] == -1:
                            label[nx, ny] = cid
                            q.append((nx, ny))
                comps.append(cells)
        return comps

    def zhang_suen_skeleton(mask: np.ndarray) -> np.ndarray:
        """
        Simple Zhang-Suen thinning. The mask is padded so that border-touching
        thick shapes can also be thinned.
        """
        H, W = mask.shape
        if not mask.any():
            return np.zeros((H, W), dtype=bool)

        X = np.pad(mask.astype(np.uint8), 2, mode="constant")
        h, w = X.shape

        while True:
            changed = False
            for step in (1, 2):
                delete = []

                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        if X[i, j] != 1:
                            continue

                        b2 = X[i, j - 1]
                        b3 = X[i + 1, j - 1]
                        b4 = X[i + 1, j]
                        b5 = X[i + 1, j + 1]
                        b6 = X[i, j + 1]
                        b7 = X[i - 1, j + 1]
                        b8 = X[i - 1, j]
                        b9 = X[i - 1, j - 1]

                        B = int(b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9)

                        A = (
                            (int(b2 == 0 and b3 == 1) +
                             int(b3 == 0 and b4 == 1) +
                             int(b4 == 0 and b5 == 1) +
                             int(b5 == 0 and b6 == 1) +
                             int(b6 == 0 and b7 == 1) +
                             int(b7 == 0 and b8 == 1) +
                             int(b8 == 0 and b9 == 1) +
                             int(b9 == 0 and b2 == 1))
                        )

                        if 2 <= B <= 6 and A == 1:
                            if step == 1 and (b2 * b4 * b6) == 0 and (b4 * b6 * b8) == 0:
                                delete.append((i, j))
                            elif step == 2 and (b2 * b4 * b8) == 0 and (b2 * b6 * b8) == 0:
                                delete.append((i, j))

                if delete:
                    for i, j in delete:
                        X[i, j] = 0
                    changed = True

            if not changed:
                break

        return (X[2:-2, 2:-2] == 1)

    def isolated_2x2_blocks(mask_color: int) -> list:
        """
        Find 2x2 uniform blocks that are not part of a larger rectangle of the
        same color. This avoids picking up large regions of a non-key color.
        """
        blocks = []
        H, W = grid.shape
        for r in range(H - 1):
            for c in range(W - 1):
                v = grid[r, c]
                if v != mask_color:
                    continue

                if not (grid[r + 1, c] == v and grid[r, c + 1] == v and grid[r + 1, c + 1] == v):
                    continue

                # Require the block to be isolated in the four cardinal directions.
                if r > 0 and grid[r - 1, c] == v:
                    continue
                if r + 2 < H and grid[r + 2, c] == v:
                    continue
                if c > 0 and grid[r, c - 1] == v:
                    continue
                if c + 2 < W and grid[r, c + 2] == v:
                    continue

                blocks.append((r, c, int(v)))
        return blocks

    # ------------------------------------------------------------------
    # Basic setup and role detection
    # ------------------------------------------------------------------
    out = np.array(grid, dtype=np.uint8, copy=True)
    H, W = out.shape
    if H < 2 or W < 2:
        return out

    # Background = most common color on the border.
    border = np.concatenate([
        out[0, :].astype(int),
        out[-1, :].astype(int),
        out[:, 0].astype(int),
        out[:, -1].astype(int)
    ])
    border_vals, border_counts = np.unique(border, return_counts=True)
    background = int(border_vals[np.argmax(border_counts)])

    if np.count_nonzero(out == background) == H * W:
        return out

    # Main color = largest connected component that is not the background.
    colors = set(int(x) for x in np.unique(out))
    main_color = -1
    best_area = 0

    for col in colors:
        if col == background:
            continue
        mask = (out == col)
        comps = components(mask)
        if comps:
            area = max(len(x) for x in comps)
            if area > best_area:
                best_area = area
                main_color = col

    if main_color == -1:
        return out

    main_mask = (out == main_color)

    # ------------------------------------------------------------------
    # Extract the repeating sequence from the top-left 2x2 key blocks.
    # ------------------------------------------------------------------
    non_main_bg_colors = colors - {background, main_color}
    all_blocks = []
    for col in non_main_bg_colors:
        all_blocks.extend(isolated_2x2_blocks(col))

    sequence = []
    marker_color = -1

    if all_blocks:
        # The top row of 2x2 key blocks is the sequence.
        top_row = min(r for r, c, v in all_blocks)
        seq_blocks = sorted([b for b in all_blocks if b[0] == top_row], key=lambda x: x[1])
        sequence = [int(v) for r, c, v in seq_blocks]

        # The remaining 2x2 block(s) provide the marker color. Prefer the lowest one.
        other_blocks = [b for b in all_blocks if b[0] != top_row]
        if other_blocks:
            marker_block = max(other_blocks, key=lambda x: (x[0], x[1]))
            marker_color = int(marker_block[2])
        else:
            # Fallback: choose a color not in sequence/background/main.
            for col in non_main_bg_colors:
                if col not in sequence:
                    marker_color = col
                    break
    else:
        # Fallback if no 2x2 blocks are found: use the two most frequent non-bg/main colors.
        freq = {col: int(np.count_nonzero(out == col)) for col in non_main_bg_colors}
        ordered = sorted(freq, key=lambda x: -freq[x])
        if ordered:
            sequence = [int(ordered[0])]
            if len(ordered) > 1:
                marker_color = int(ordered[1])

    if not sequence:
        return out

    if marker_color == -1:
        # Last resort: use the least common non-background/main color.
        freq = {col: int(np.count_nonzero(out == col)) for col in non_main_bg_colors}
        if freq:
            marker_color = int(min(freq, key=lambda x: freq[x]))
        else:
            return out

    # ------------------------------------------------------------------
    # Anchor pixels: single pixels of the first sequence color on the left border.
    # ------------------------------------------------------------------
    first_color = int(sequence[0])
    anchors = [(r, 0) for r in range(H) if out[r, 0] == first_color]

    # If no exact left-border anchors exist, try the first two columns.
    if not anchors:
        for r in range(H):
            for c in range(0, min(2, W)):
                if out[r, c] == first_color:
                    anchors.append((r, c))
                    break

    if not anchors:
        return out

    # ------------------------------------------------------------------
    # Build the thin guide: skeleton of the main thick region + anchor attachments.
    # ------------------------------------------------------------------
    skeleton = zhang_suen_skeleton(main_mask)
    skel_cells = set(map(tuple, np.argwhere(skeleton)))

    path_cells = set(skel_cells)
    anchor_set = set(anchors)
    path_cells.update(anchor_set)

    # Connect each anchor to the nearest skeleton cell, preferring the same row.
    for r, c in anchors:
        same_row = [p for p in skel_cells if p[0] == r and p[1] > c]
        if same_row:
            tr, tc = min(same_row, key=lambda p: p[1])
            for cc in range(c, tc + 1):
                path_cells.add((r, cc))
        elif skel_cells:
            tr, tc = min(skel_cells, key=lambda p: (abs(p[0] - r) + abs(p[1] - c), p[0], p[1]))
            # Simple L-shaped attachment.
            for cc in range(c, tc + 1):
                path_cells.add((r, cc))
            for rr in range(min(r, tr), max(r, tr) + 1):
                path_cells.add((rr, tc))

    # Keep only path components that contain an anchor.
    path_mask = np.zeros((H, W), dtype=bool)
    for r, c in path_cells:
        if 0 <= r < H and 0 <= c < W:
            path_mask[r, c] = True

    path_comps = components(path_mask)
    path_comps = [comp for comp in path_comps if any(p in anchor_set for p in comp)]

    if not path_comps:
        return out

    # ------------------------------------------------------------------
    # Color each anchor-connected path component.
    # ------------------------------------------------------------------
    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    L = len(sequence)

    for comp in path_comps:
        comp_set = set(comp)

        # Neighbors within this component.
        neigh = {}
        for r, c in comp:
            ns = []
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in comp_set:
                    ns.append((nr, nc))
            neigh[(r, c)] = ns

        # Start from the lowest anchor in this component.
        comp_anchors = [p for p in comp if p in anchor_set]
        start = max(comp_anchors, key=lambda p: (p[0], p[1]))

        # BFS distance from the start, used for the repeating sequence.
        dist = {start: 0}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for nx, ny in neigh[(x, y)]:
                if (nx, ny) not in dist:
                    dist[(nx, ny)] = dist[(x, y)] + 1
                    q.append((nx, ny))

        # Paint the sequence.
        for (r, c), d in dist.items():
            out[r, c] = sequence[d % L]

        # Place marker color just outside the main region at terminal path cells.
        for p, ns in neigh.items():
            if len(ns) != 1:
                continue
            if p in anchor_set:
                continue

            r, c = p
            nr0, nc0 = ns[0]
            dr = r - nr0
            dc = c - nc0

            rr, cc = r, c
            while True:
                rr += dr
                cc += dc
                if not (0 <= rr < H and 0 <= cc < W):
                    break

                # If we are still on the path, continue.
                if (rr, cc) in comp_set:
                    continue

                # The first non-main cell in this direction is the marker position.
                if not main_mask[rr, cc]:
                    out[rr, cc] = marker_color
                    break
                # If the direction initially goes through main, keep going until outside.
                else:
                    continue

    return out