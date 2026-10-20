import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed empty cells with the "key" color.

    Rule inferred from the examples:
      1. The background color is the most common color.
      2. The fill/key color is the rarest non-background color.  If several
         colors tie, prefer colors on the border and break remaining ties by
         choosing the largest color value.
      3. Treat every non-background cell as a "wall"/structure cell.
         Structure cells are connected using 8-connectivity.
      4. The key cell belongs to one 8-connected structure component.
      5. Find all 4-connected background components.  If a background component
         does not touch the grid border and is orthogonally adjacent to the
         key's structure component, fill that whole background component with
         the key color.

    This fills the interiors of closed rooms that are connected to the key by
    the wall structure, while leaving isolated rooms connected to other wall
    components unchanged.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = grid.copy()

    # ------------------------------------------------------------------
    # 1. Identify background color.
    # ------------------------------------------------------------------
    vals, counts_arr = np.unique(out, return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals, counts_arr)}

    if len(counts) <= 1:
        return out

    max_count = max(counts.values())
    bg_candidates = [c for c, n in counts.items() if n == max_count]
    if 0 in bg_candidates:
        bg = 0
    else:
        bg = min(bg_candidates)

    non_bg_counts = {c: n for c, n in counts.items() if c != bg}
    if not non_bg_counts:
        return out

    # ------------------------------------------------------------------
    # 2. Identify the key/fill color.
    #    Prefer rarest, then border occurrence, then largest color value.
    # ------------------------------------------------------------------
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    border_colors = set()
    border_vals = np.unique(out[border_mask])
    for v in border_vals:
        iv = int(v)
        if iv != bg:
            border_colors.add(iv)

    min_count = min(non_bg_counts.values())
    candidates = [c for c, n in non_bg_counts.items() if n == min_count]

    border_candidates = [c for c in candidates if c in border_colors]
    if border_candidates:
        candidates = border_candidates

    key_color = max(candidates)

    # ------------------------------------------------------------------
    # 3. Label 8-connected structure components (all non-background cells).
    # ------------------------------------------------------------------
    structure = (out != bg)
    comp_id = -np.ones((H, W), dtype=int)
    comp_index = 0

    for r in range(H):
        for c in range(W):
            if not structure[r, c] or comp_id[r, c] != -1:
                continue

            q = deque([(r, c)])
            comp_id[r, c] = comp_index
            comp_index += 1

            while q:
                cr, cc = q.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < H and 0 <= nc < W and
                            structure[nr, nc] and comp_id[nr, nc] == -1
                        ):
                            comp_id[nr, nc] = comp_index
                            q.append((nr, nc))

    # ------------------------------------------------------------------
    # 4. Find the structure component containing the key.
    # ------------------------------------------------------------------
    key_cells = []
    for r in range(H):
        for c in range(W):
            if out[r, c] == key_color:
                key_cells.append((r, c))

    if not key_cells:
        return out

    key_comp_set = set(comp_id[r, c] for r, c in key_cells)

    # If the key color occurs in multiple structure components, choose one
    # deterministically: prefer a component with a key cell on the border,
    # then the component containing the most key cells, then smallest id.
    if len(key_comp_set) == 1:
        key_comp = key_comp_set.pop()
    else:
        key_counts = {}
        for r, c in key_cells:
            cid = comp_id[r, c]
            key_counts[cid] = key_counts.get(cid, 0) + 1

        border_comps = {
            comp_id[r, c] for r, c in key_cells
            if r == 0 or r == H - 1 or c == 0 or c == W - 1
        }

        if border_comps:
            selected = [cid for cid in key_comp_set if cid in border_comps]
        else:
            selected = list(key_comp_set)

        key_comp = min(selected, key=lambda cid: (-key_counts[cid], cid))

    key_comp_mask = (comp_id == key_comp)

    # ------------------------------------------------------------------
    # 5. Find 4-connected background components and fill eligible holes.
    # ------------------------------------------------------------------
    bg_mask = (out == bg)
    visited = np.zeros((H, W), dtype=bool)

    for sr in range(H):
        for sc in range(W):
            if not bg_mask[sr, sc] or visited[sr, sc]:
                continue

            q = deque([(sr, sc)])
            visited[sr, sc] = True
            cells = []
            touches_border = False

            while q:
                r, c = q.popleft()
                cells.append((r, c))

                if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                    touches_border = True

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < H and 0 <= nc < W and
                        bg_mask[nr, nc] and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))

            # Only fill holes, not the outside background.
            if touches_border:
                continue

            # The hole must touch the key's structure component orthogonally.
            fill = False
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < H and 0 <= nc < W and
                        structure[nr, nc] and key_comp_mask[nr, nc]
                    ):
                        fill = True
                        break
                if fill:
                    break

            if fill:
                for r, c in cells:
                    out[r, c] = key_color

    return out