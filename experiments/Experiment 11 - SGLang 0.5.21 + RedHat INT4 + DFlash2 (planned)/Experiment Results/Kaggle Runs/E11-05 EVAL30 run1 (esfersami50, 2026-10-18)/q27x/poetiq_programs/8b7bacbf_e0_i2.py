import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill empty enclosed rooms whose boundary is touched by the "guide" color.

    Rule inferred from the examples:
      - The background is the most frequent color.
      - The "paint" color is the rarest non-background color.  If several colors
        tie for rarest, choose the largest one.  In the examples this is the
        unique 7 or 4 cell.
      - The "guide" color is the non-background, non-paint color adjacent to the
        paint cell.  If multiple guide colors are adjacent, choose the one with
        the most pixels; ties are broken by the larger color value.
      - Find all 4-connected background components.  A component is filled if:
          * it does not touch the grid border, and
          * at least one cell in the component is adjacent (4-neighbour) to a
            guide-coloured pixel.
      - The filled component is changed to the paint colour.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    out = np.array(grid, copy=True)

    # Count colours.
    colors = np.unique(grid)
    counts = {int(c): int(np.sum(grid == c)) for c in colors}

    # Background: most common colour.
    bg_color = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # Paint colour: rarest non-background colour; largest on ties.
    nonbg_counts = {c: n for c, n in counts.items() if c != bg_color}
    if not nonbg_counts:
        return out

    paint_color = min(nonbg_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # Locate all paint cells.  If there is one, use it.  If several, use all
    # border cells adjacent to any paint cell for guide-colour selection.
    paint_cells = np.argwhere(grid == paint_color)
    if len(paint_cells) == 0:
        return out

    # Neighbour colours adjacent to any paint cell.
    adj_colors = {}
    for r, c in paint_cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                v = int(grid[nr, nc])
                if v != bg_color and v != paint_color:
                    adj_colors[v] = adj_colors.get(v, 0) + 1

    if not adj_colors:
        return out

    guide_color = max(adj_colors.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # Mark guide pixels for quick lookup.
    guide_mask = (grid == guide_color)

    # 4-connected components of background cells.
    seen = np.zeros((H, W), dtype=bool)

    for sr in range(H):
        for sc in range(W):
            if seen[sr, sc] or grid[sr, sc] != bg_color:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp = [(sr, sc)]

            touches_border = (sr == 0 or sc == 0 or
                              sr == H - 1 or sc == W - 1)

            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < H and 0 <= nc < W and
                            not seen[nr, nc] and
                            grid[nr, nc] == bg_color):
                        seen[nr, nc] = True
                        comp.append((nr, nc))
                        q.append((nr, nc))
                        if nr == 0 or nc == 0 or nr == H - 1 or nc == W - 1:
                            touches_border = True

            # Outside components are not filled.
            if touches_border:
                continue

            # Does this hole touch the guide colour?
            for r, c in comp:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and guide_mask[nr, nc]:
                        for rr, cc in comp:
                            out[rr, cc] = paint_color
                        break
                else:
                    continue
                break

    return out