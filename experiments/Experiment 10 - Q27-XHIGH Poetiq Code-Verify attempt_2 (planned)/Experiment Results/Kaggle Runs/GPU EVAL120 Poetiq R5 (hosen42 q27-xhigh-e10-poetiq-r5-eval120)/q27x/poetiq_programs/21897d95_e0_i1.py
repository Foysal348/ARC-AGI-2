import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input mosaic into the solved mosaic.

    The input is interpreted as a grid of rectangular tiles.  Some tiles contain
    small 1-patterns that act as “paths”.  Each such tile has a source color:
      - if the 1-pattern surrounds a single non-1 cell, that cell's color is used;
      - otherwise the dominant non-1 color of the tile is used.

    The path paints other tiles with that source color.  Paths are processed in
    descending source-color order and stop when they hit an already-painted tile.

    The output is the painted tile matrix, rendered as a solid-color image.  If
    the natural tile matrix is taller than wide, it is transposed before painting;
    this matches the non-square examples.
    """
    if grid.size == 0:
        return np.asarray(grid, dtype=int)

    H, W = grid.shape

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def clean_mask(g: np.ndarray) -> np.ndarray:
        return g != 1

    def dominant_color(g: np.ndarray) -> int:
        mask = clean_mask(g)
        if not mask.any():
            return 1
        vals = g[mask]
        vals = vals[vals != 1]
        if vals.size == 0:
            return 1
        vals, counts = np.unique(vals, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def tile_purity(g: np.ndarray) -> float:
        mask = clean_mask(g)
        if not mask.any():
            return 1.0
        d = dominant_color(g)
        return float(np.sum((g == d) & mask)) / float(np.sum(mask))

    def line_scores(axis: int) -> np.ndarray:
        """
        Score possible tile boundaries.
        axis=0: horizontal lines between rows.
        axis=1: vertical lines between columns.
        """
        if axis == 0:
            n = H - 1
            if n <= 0:
                return np.zeros(0, dtype=int)
            # Count changes that are not caused only by 1s.
            diff = (grid[:-1, :] != grid[1:, :]) & ~((grid[:-1, :] == 1) & (grid[1:, :] == 1))
            return diff.sum(axis=1)
        else:
            n = W - 1
            if n <= 0:
                return np.zeros(0, dtype=int)
            diff = (grid[:, :-1] != grid[:, 1:]) & ~((grid[:, :-1] == 1) & (grid[:, 1:] == 1))
            return diff.sum(axis=0)

    def choose_boundaries(scores: np.ndarray, length: int, max_lines: int = 5) -> list:
        """
        Choose a small set of plausible boundary positions.
        """
        if scores.size == 0 or length <= 1:
            return []
        max_score = int(scores.max())
        if max_score <= 0:
            return []
        min_score = max(3, int(0.18 * max_score))
        cand = np.where(scores >= min_score)[0]
        if cand.size == 0:
            top = np.argsort(scores)[::-1][:max_lines]
            cand = top
        # Keep the strongest, avoiding adjacent duplicates.
        selected = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] < min_score:
                break
            if any(abs(i - j) <= 1 for j in selected):
                continue
            selected.append(i)
            if len(selected) >= max_lines:
                break
        selected.sort()
        return selected

    h_bounds = [0] + [i + 1 for i in choose_boundaries(line_scores(0), H)] + [H]
    v_bounds = [0] + [i + 1 for i in choose_boundaries(line_scores(1), W)] + [W]

    # If segmentation collapses, fall back to one tile.
    if h_bounds[-1] != H:
        h_bounds = [0, H]
    if v_bounds[-1] != W:
        v_bounds = [0, W]

    R = len(h_bounds) - 1
    C = len(v_bounds) - 1

    # If the detected grid is taller than wide, transpose the working matrix.
    work_grid = grid.copy()
    if R > C:
        work_grid = np.ascontiguousarray(work_grid.T)
        h_bounds, v_bounds = v_bounds, h_bounds
        R, C = C, R
        work_H, work_W = work_grid.shape

    # ------------------------------------------------------------------
    # Extract tiles and sources
    # ------------------------------------------------------------------
    base = np.full((R, C), -1, dtype=int)
    sources = []  # list of dicts: r, c, color, cells (1-mask in tile)

    for i in range(R):
        r0, r1 = h_bounds[i], h_bounds[i + 1]
        for j in range(C):
            c0, c1 = v_bounds[j], v_bounds[j + 1]
            tile = work_grid[r0:r1, c0:c1]
            base[i, j] = dominant_color(tile)

            mask = (tile == 1)
            if not mask.any():
                continue

            # Largest connected component of 1s is the glyph.
            comp = np.zeros_like(mask)
            visited = np.zeros_like(mask)
            best_size = 0
            h, w = tile.shape
            for r in range(h):
                for c in range(w):
                    if mask[r, c] and not visited[r, c]:
                        stack = [(r, c)]
                        comp_idx = []
                        visited[r, c] = True
                        while stack:
                            y, x = stack.pop()
                            comp_idx.append((y, x))
                            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    if mask[ny, nx] and not visited[ny, nx]:
                                        visited[ny, nx] = True
                                        stack.append((ny, nx))
                        if len(comp_idx) > best_size:
                            best_size = len(comp_idx)
                            comp[:] = False
                            for y, x in comp_idx:
                                comp[y, x] = True

            if not comp.any():
                continue

            # Detect a special centered non-1 cell surrounded by 1s.
            special_color = None
            special_pos = None
            for r in range(h):
                for c in range(w):
                    if tile[r, c] == 1:
                        continue
                    n_up = r > 0 and comp[r - 1, c]
                    n_down = r + 1 < h and comp[r + 1, c]
                    n_left = c > 0 and comp[r, c - 1]
                    n_right = c + 1 < w and comp[r, c + 1]
                    if int(n_up) + int(n_down) + int(n_left) + int(n_right) >= 3:
                        special_color = int(tile[r, c])
                        special_pos = (r, c)
                        break
                if special_color is not None:
                    break

            if special_color is not None:
                color = special_color
            else:
                color = base[i, j]

            sources.append({"r": i, "c": j, "color": color, "comp": comp, "shape": tile.shape,
                            "special_pos": special_pos})

    # ------------------------------------------------------------------
    # Infer path template from each source glyph
    # ------------------------------------------------------------------
    def directions_from_special(comp: np.ndarray, pos):
        """
        For a 3-side T around a special center, return the unique present direction.
        """
        h, w = comp.shape
        r, c = pos
        dirs = set()
        if r > 0 and comp[r - 1, c]:
            dirs.add('U')
        if r + 1 < h and comp[r + 1, c]:
            dirs.add('D')
        if c > 0 and comp[r, c - 1]:
            dirs.add('L')
        if c + 1 < w and comp[r, c + 1]:
            dirs.add('R')

        if 'U' in dirs and 'D' in dirs:
            if 'L' in dirs:
                return 'L'
            if 'R' in dirs:
                return 'R'
        if 'L' in dirs and 'R' in dirs:
            if 'U' in dirs:
                return 'U'
            if 'D' in dirs:
                return 'D'
        # Fallback: first unique or arbitrary.
        for d in ['L', 'R', 'U', 'D']:
            if d in dirs:
                return d
        return 'D'

    def horizontal_branch(comp: np.ndarray):
        """
        Detect an L-shaped glyph with a vertical shaft and a left/right branch.
        Returns 'L' or 'R' if clear, else None.
        """
        h, w = comp.shape
        rows = np.where(comp.any(axis=1))[0]
        cols = np.where(comp.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            return None

        # Main vertical shaft: column with the most 1s.
        col_counts = comp.sum(axis=0)
        main_c = int(np.argmax(col_counts))

        left_branch = False
        right_branch = False
        for r in range(h):
            if comp[r, main_c]:
                if c0 := max(0, main_c - 2):
                    if comp[r, c0] and c0 < main_c:
                        left_branch = True
                if main_c + 1 < w and comp[r, main_c + 1]:
                    right_branch = True

        if left_branch and not right_branch:
            return 'L'
        if right_branch and not left_branch:
            return 'R'
        return None

    def vertical_arrow(comp: np.ndarray):
        """
        Detect a simple up/down arrow glyph.
        """
        h, w = comp.shape
        rows = np.where(comp.any(axis=1))[0]
        if rows.size == 0:
            return None
        top = rows[0]
        bottom = rows[-1]
        mid = rows[len(rows) // 2]

        # If there are 1s near both top and bottom but not strongly horizontal,
        # consider it bidirectional vertical.
        top_mass = comp[top:top + max(1, h // 4), :].sum()
        bot_mass = comp[max(0, bottom - max(1, h // 4) + 1):bottom + 1, :].sum()
        if top_mass > 0 and bot_mass > 0:
            return 'UD'

        # Prefer the side with more mass.
        if top_mass < bot_mass:
            return 'D'
        if bot_mass < top_mass:
            return 'U'
        return None

    # ------------------------------------------------------------------
    # Generate path cells for a source, and paint
    # ------------------------------------------------------------------
    painted = np.full((R, C), -1, dtype=int)

    # Process higher colors first; this matches the examples.
    sources.sort(key=lambda x: (-x["color"], x["r"], x["c"]))

    for s in sources:
        color = s["color"]
        sr, sc = s["r"], s["c"]
        comp = s["comp"]
        special = s["special_pos"] is not None

        if special:
            f = directions_from_special(comp, s["special_pos"])
            if f in ('U', 'D'):
                # Straight vertical path.
                dr = -1 if f == 'U' else 1
                r, c = sr + dr, sc
                cells = []
                while 0 <= r < R and 0 <= c < C:
                    cells.append((r, c))
                    r += dr
            else:
                # Horizontal first, then vertical turn, then horizontal again.
                dc = -1 if f == 'L' else 1
                # For left-turn glyphs use up, for right-turn glyphs use down.
                dr = -1 if f == 'L' else 1
                r, c = sr, sc + dc
                cells = []
                # First horizontal step.
                if 0 <= r < R and 0 <= c < C:
                    cells.append((r, c))
                    # Vertical to border.
                    r += dr
                    while 0 <= r < R and 0 <= c < C:
                        cells.append((r, c))
                        r += dr
                    # Horizontal again.
                    r = sr + sum(d for d in [] )  # not used, reset below
                    r, c = sr + (dr * 0), sc  # placeholder, recompute from last valid
                # Recompute robustly:
                r, c = sr, sc + dc
                cells = []
                if 0 <= r < R and 0 <= c < C:
                    cells.append((r, c))
                    r += dr
                    while 0 <= r < R and 0 <= c < C:
                        cells.append((r, c))
                        r += dr
                    # Return to the corner and continue horizontally.
                    r, c = sr + (0 if False else 0), sc
                    # The corner is after vertical movement; use last valid vertical endpoint.
                    # Recompute endpoint:
                    rr, cc = sr, sc + dc
                    if 0 <= rr < R and 0 <= cc < C:
                        rr += dr
                        last_rr, last_cc = sr, sc + dc
                        while 0 <= rr < R and 0 <= cc < C:
                            last_rr, last_cc = rr, cc
                            rr += dr
                        r, c = last_rr, last_cc + dc
                        while 0 <= r < R and 0 <= c < C:
                            cells.append((r, c))
                            c += dc
        else:
            h = horizontal_branch(comp)
            if h in ('L', 'R'):
                # Background L glyph: horizontal step, then down, then horizontal again.
                dc = -1 if h == 'L' else 1
                dr = 1  # examples use downward vertical turn for background glyphs
                r, c = sr, sc + dc
                cells = []
                if 0 <= r < R and 0 <= c < C:
                    cells.append((r, c))
                    r += dr
                    last_rr, last_cc = sr, sc + dc
                    while 0 <= r < R and 0 <= c < C:
                        last_rr, last_cc = r, c
                        cells.append((r, c))
                        r += dr
                    r, c = last_rr, last_cc + dc
                    while 0 <= r < R and 0 <= c < C:
                        cells.append((r, c))
                        c += dc
            else:
                v = vertical_arrow(comp)
                if v == 'UD':
                    cells = []
                    # Up
                    r = sr - 1
                    while 0 <= r < R and 0 <= sc < C:
                        cells.append((r, sc))
                        r -= 1
                    # Down
                    r = sr + 1
                    while 0 <= r < R and 0 <= sc < C:
                        cells.append((r, sc))
                        r += 1
                elif v in ('U', 'D'):
                    dr = -1 if v == 'U' else 1
                    r, c = sr + dr, sc
                    cells = []
                    while 0 <= r < R and 0 <= c < C:
                        cells.append((r, c))
                        r += dr
                else:
                    cells = []

        # Paint path, stopping at already painted cells.
        for r, c in cells:
            if 0 <= r < R and 0 <= c < C and painted[r, c] == -1:
                painted[r, c] = color
            else:
                # Stop this segment/path at the first blocked cell.
                break

    # Any unpainted tile keeps its base color.
    for i in range(R):
        for j in range(C):
            if painted[i, j] == -1:
                painted[i, j] = base[i, j]

    # ------------------------------------------------------------------
    # Render painted tile matrix back to an image.
    # ------------------------------------------------------------------
    out = np.empty((work_H, work_W), dtype=grid.dtype)
    for i in range(R):
        r0, r1 = h_bounds[i], h_bounds[i + 1]
        for j in range(C):
            c0, c1 = v_bounds[j], v_bounds[j + 1]
            out[r0:r1, c0:c1] = painted[i, j]

    return out