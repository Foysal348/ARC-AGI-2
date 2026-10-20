import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Repair corrupted repeating patterns inside framed panels.

    The task contains rectangular panels separated/framed by a border color.
    Inside each panel there is a small periodic pattern made from a background
    color and one (or more) foreground colors.  A few cells are corrupted.
    The transformation finds each panel, infers its small periodic tile, and
    reconstructs the panel from that tile.
    """
    if grid.size == 0:
        return grid.copy()

    grid = grid.astype(np.int8).copy()
    h, w = grid.shape

    # The outer background color is the color of the outer corner.
    outer_color = int(grid[0, 0])

    # The panel frame color is normally the cell just inside the outer frame.
    # If that fails, try the next most plausible interior border color.
    frame_color = None
    if h > 1 and w > 1:
        candidate = int(grid[1, 1])
        if candidate != outer_color:
            frame_color = candidate
        else:
            # Fallback: look for a color adjacent to the top/left edge.
            edge_colors = []
            for i in range(1, min(h, 4)):
                for j in range(1, min(w, 4)):
                    c = int(grid[i, j])
                    if c != outer_color and c not in edge_colors:
                        edge_colors.append(c)
            if edge_colors:
                frame_color = edge_colors[0]

    if frame_color is None:
        # Nothing recognizable to repair; return unchanged.
        return grid

    # Work on a copy.
    out = grid.copy()

    # Panels are connected components of cells that are not the frame color.
    # The component touching the outer border is the exterior background.
    mask = grid != frame_color

    def component_cells(start_r, start_c):
        if start_r < 0 or start_r >= h or start_c < 0 or start_c >= w:
            return []
        if not mask[start_r, start_c]:
            return []

        seen = np.zeros((h, w), dtype=bool)
        q = deque([(start_r, start_c)])
        seen[start_r, start_c] = True
        cells = []

        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        return cells

    def touches_border(cells):
        for r, c in cells:
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                return True
        return False

    visited = np.zeros((h, w), dtype=bool)
    panels = []

    for r in range(h):
        for c in range(w):
            if not visited[r, c] and mask[r, c]:
                visited[r, c] = True
                cells = component_cells(r, c)
                for rr, cc in cells:
                    visited[rr, cc] = True
                if not touches_border(cells):
                    panels.append(cells)

    if not panels:
        return out

    # Infer a small periodic tile for a panel.
    def infer_tile(panel: np.ndarray):
        ph, pw = panel.shape
        if ph == 0 or pw == 0:
            return panel.copy()

        # ARC patterns are small; search modest periods.
        max_period = 12
        best_tile = panel.copy()
        best_score = None
        best_area = ph * pw

        present_colors = np.unique(panel)
        for cand_h in range(1, min(ph, max_period) + 1):
            for cand_w in range(1, min(pw, max_period) + 1):
                # Avoid choosing the whole panel as the tile.
                if cand_h == ph and cand_w == pw:
                    continue

                # Count, for each tile position, how often each color appears.
                counts = {}
                for r in range(ph):
                    for c in range(pw):
                        counts[(r % cand_h, c % cand_w)] = {}

                for r in range(ph):
                    rr = r % cand_h
                    row = panel[r]
                    for c in range(pw):
                        cc = c % cand_w
                        col = int(row[c])
                        key = (rr, cc)
                        counts[key][col] = counts[key].get(col, 0) + 1

                tile = np.empty((cand_h, cand_w), dtype=np.int8)
                for r in range(cand_h):
                    for c in range(cand_w):
                        vals = counts[(r, c)]
                        # Tie-break: prefer the outer/background color, then lower id.
                        best = None
                        best_count = -1
                        for color, cnt in vals.items():
                            if (cnt, -int(color == outer_color), -color) > \
                               (best_count, -int(best == outer_color), -best if best is not None else 10**9):
                                best = color
                                best_count = cnt
                        tile[r, c] = best if best is not None else int(outer_color)

                # Fit: how many panel cells are already consistent with tile.
                tiled = np.empty((ph, pw), dtype=np.int8)
                for r in range(ph):
                    for c in range(pw):
                        tiled[r, c] = tile[r % cand_h, c % cand_w]

                fit = int(np.sum(tiled == panel))
                area = cand_h * cand_w

                # Slightly prefer smaller tiles among similarly good fits.
                score = fit - 0.10 * area

                if best_score is None or score > best_score + 1e-9:
                    best_score = score
                    best_tile = tile
                    best_area = area
                elif abs(score - best_score) <= 1e-9:
                    if area < best_area:
                        best_score = score
                        best_tile = tile
                        best_area = area

        return best_tile

    # Fill each panel with its inferred repeated tile.
    for cells in panels:
        if not cells:
            continue

        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r0, r1 = min(rs), max(rs) + 1
        c0, c1 = min(cs), max(cs) + 1

        panel = out[r0:r1, c0:c1]
        tile = infer_tile(panel)
        th, tw = tile.shape

        filled = np.empty((r1 - r0, c1 - c0), dtype=np.int8)
        for i in range(r1 - r0):
            for j in range(c1 - c0):
                filled[i, j] = tile[i % th, j % tw]

        # Update only the panel cells, not any frame pixels accidentally in bbox.
        panel_mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
        for r, c in cells:
            panel_mask[r - r0, c - c0] = True

        # If the panel is effectively a solid rectangle, fill all of it.
        if panel_mask.all():
            out[r0:r1, c0:c1] = filled
        else:
            out[r0:r1, c0:c1][panel_mask] = filled[panel_mask]

    return out