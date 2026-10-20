import numpy as np
from itertools import combinations
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input by finding the small square "frame" panels, extracting
    their interiors, undoing random symmetries of the panel contents, and then
    XOR-ing the colored pixels parity-wise.

    The final output is:
      - 0 when no color has odd parity at that cell, or when multiple colors
        have odd parity at the same cell;
      - 6 when exactly color 6 has odd parity;
      - 2 when exactly one non-6 color has odd parity.

    The algorithm:
      1. Detect all plausible closed uniform rectangular frames.
      2. Keep candidates with small, mostly-zero interiors that do not contain
         the frame border color.
      3. Choose the most consistent interior shape (the task examples use four
         square panels of the same size).
      4. Search over symmetries of the chosen panels and over small subsets of
         the top candidates.
      5. Choose the subset + symmetry combination whose XOR result is most
         "panel-like": compact, mostly line-shaped, not overly dense, and with
         any color-6 pixel well attached to the rest of the shape.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # ------------------------------------------------------------------
    # Helper: score a candidate frame rectangle.
    # Lower is better.
    # ------------------------------------------------------------------
    def rectangle_stats(r0, c0, r1, c1, color):
        if r1 - r0 < 2 or c1 - c0 < 2:
            return None

        interior = grid[r0 + 1:r1, c0 + 1:c1]
        if interior.size == 0:
            return None

        area = interior.size
        border_inside = int(np.sum(interior == color))
        zero_frac = float(np.sum(interior == 0)) / area
        distinct = len(np.unique(interior))
        side_r = r1 - r0
        side_c = c1 - c0

        penalty = 0.0

        # The intended panel interiors are sparse and do not contain the border.
        if border_inside:
            penalty += 1000.0 * border_inside

        if zero_frac < 0.15:
            penalty += 1000.0 * (0.15 - zero_frac)

        # Intended interiors are small and usually square.
        if side_r != side_c:
            penalty += 25.0 * abs(side_r - side_c)

        max_side = max(side_r, side_c)
        if max_side > 12:
            penalty += 35.0 * (max_side - 12)

        # Interior color variety is usually tiny: 0 plus a few signal colors.
        if distinct > 6:
            penalty += 30.0 * (distinct - 6)

        # Prefer small, clean panels.
        penalty += 0.01 * (side_r + side_c)

        return {
            "penalty": penalty,
            "shape": (side_r - 1, side_c - 1),
            "border": int(color),
            "rect": (r0, c0, r1, c1),
        }

    # ------------------------------------------------------------------
    # Detect candidate frames.
    # ------------------------------------------------------------------
    candidates = []
    colors = np.unique(grid)

    # ARC grids are small; this limit is safe and avoids huge background frames.
    max_side = min(18, max(h, w))

    for color in colors:
        if color == 0:
            continue

        for r0 in range(h):
            for c0 in range(w):
                if grid[r0, c0] != color:
                    continue

                # Try possible right edges.
                for c1 in range(c0 + 2, min(w, c0 + max_side + 1)):
                    if grid[r0, c1] != color:
                        continue
                    if not np.all(grid[r0, c0:c1 + 1] == color):
                        continue

                    # Try possible bottom edges.
                    for r1 in range(r0 + 2, min(h, r0 + max_side + 1)):
                        if grid[r1, c0] != color or grid[r1, c1] != color:
                            continue
                        if not np.all(grid[r1, c0:c1 + 1] == color):
                            continue
                        if not np.all(grid[r0:r1 + 1, c0] == color):
                            continue
                        if not np.all(grid[r0:r1 + 1, c1] == color):
                            continue

                        stats = rectangle_stats(r0, c0, r1, c1, color)
                        if stats is None:
                            continue

                        # Ignore clearly bad rectangles early.
                        if stats["penalty"] > 500:
                            continue

                        candidates.append(stats)

    if not candidates:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # Choose the most common / most plausible interior shape.
    # ------------------------------------------------------------------
    shape_counter = Counter(cand["shape"] for cand in candidates)

    # Prefer shapes that appear at least twice; if none, use the best single one.
    viable = {sh: cnt for sh, cnt in shape_counter.items() if cnt >= 2}
    if viable:
        target_shape = max(viable.items(), key=lambda kv: (kv[1], -sum(c["penalty"] for c in candidates if c["shape"] == kv[0])))
        target_shape = target_shape[0]
    else:
        candidates.sort(key=lambda x: x["penalty"])
        target_shape = candidates[0]["shape"]

    shape_cands = [c for c in candidates if c["shape"] == target_shape]
    shape_cands.sort(key=lambda x: x["penalty"])

    # Keep a small pool for robust subset selection.
    pool = shape_cands[:8]

    # ------------------------------------------------------------------
    # Convert an interior array into a uint32 bitmask per cell.
    # Bit c is set if color c is present.
    # ------------------------------------------------------------------
    def interior_to_mask(arr):
        mask = np.zeros(arr.shape, dtype=np.uint32)
        for c in np.unique(arr):
            if c == 0:
                continue
            mask[arr == int(c)] |= np.uint32(1 << int(c))
        return mask

    def symmetries(arr):
        """
        Return the 8 dihedral symmetries of a square array as bitmask arrays.
        If the array is not square, only return identity/horizontal/vertical/180.
        """
        if arr.shape[0] != arr.shape[1]:
            return [
                interior_to_mask(arr),
                interior_to_mask(arr[:, ::-1]),
                interior_to_mask(arr[::-1, :]),
                interior_to_mask(arr[::-1, ::-1]),
            ]

        base = arr
        variants = [
            base,
            base[:, ::-1],
            base[::-1, :],
            base[::-1, ::-1],
            base.T,
            base.T[:, ::-1],
            base.T[::-1, :],
            base.T[::-1, ::-1],
        ]
        return [interior_to_mask(v) for v in variants]

    def prepare_panels(indices):
        panels = []
        for cand in pool[indices]:
            r0, c0, r1, c1 = cand["rect"]
            interior = grid[r0 + 1:r1, c0 + 1:c1]
            panels.append(symmetries(interior))
        return panels

    # ------------------------------------------------------------------
    # Mapping from bitmask to output color.
    # Bit 6 is special; any other single odd color becomes 2.
    # Multiple odd colors cancel to 0.
    # ------------------------------------------------------------------
    color_map = np.zeros(1 << 10, dtype=np.uint8)
    for m in range(1, 1 << 10):
        # popcount
        x = m
        pc = 0
        while x:
            x &= x - 1
            pc += 1
        if pc == 1:
            # extract color
            color = (m & -m).bit_length()  # 1<<k -> k+1? check below
            # Python: (1<<c).bit_length() = c+1, so subtract 1.
            color -= 1
            if color == 6:
                color_map[m] = 6
            else:
                color_map[m] = 2

    # ------------------------------------------------------------------
    # Connected components (4-connected) for occupied cells.
    # ------------------------------------------------------------------
    def count_components(mask):
        if not mask.any():
            return 0
        visited = np.zeros_like(mask, dtype=bool)
        comps = 0
        hmask, wmask = mask.shape
        for i in range(hmask):
            for j in range(wmask):
                if mask[i, j] and not visited[i, j]:
                    comps += 1
                    stack = [(i, j)]
                    visited[i, j] = True
                    while stack:
                        x, y = stack.pop()
                        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                            if 0 <= nx < hmask and 0 <= ny < wmask:
                                if mask[nx, ny] and not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    stack.append((nx, ny))
        return comps

    # ------------------------------------------------------------------
    # Score a final output array.
    # ------------------------------------------------------------------
    def score_output(out, has_six_input):
        occupied = out != 0
        n = int(occupied.sum())
        if n == 0:
            return -100.0

        area = out.size
        density = n / area

        # Density: avoid almost-empty or almost-full results.
        density_score = 1.0
        if density > 0.85:
            density_score -= 2.0 * (density - 0.85)
        if density < 0.08:
            density_score -= 1.0 * (0.08 - density)

        # Line-likeness: occupied cells should have at least one orthogonal neighbor.
        line_cells = 0
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                if occupied[i, j]:
                    if (j > 0 and occupied[i, j - 1]) or (j + 1 < out.shape[1] and occupied[i, j + 1]) or \
                       (i > 0 and occupied[i - 1, j]) or (i + 1 < out.shape[0] and occupied[i + 1, j]):
                        line_cells += 1
        line_ratio = line_cells / n

        comps = count_components(occupied)

        # Six handling: if the panels contained 6, prefer a visible, attached 6.
        six_score = 1.0
        if has_six_input:
            sixes = np.argwhere(out == 6)
            if len(sixes) == 0:
                six_score = 0.0
            else:
                best_adj = 0
                for i, j in sixes:
                    adj = 0
                    if i > 0 and out[i - 1, j] != 0:
                        adj += 1
                    if i + 1 < out.shape[0] and out[i + 1, j] != 0:
                        adj += 1
                    if j > 0 and out[i, j - 1] != 0:
                        adj += 1
                    if j + 1 < out.shape[1] and out[i, j + 1] != 0:
                        adj += 1
                    best_adj = max(best_adj, adj)
                six_score = min(1.0, best_adj / 2.0)

        comp_penalty = 0.10 * max(0, comps - 1)

        return 3.0 * line_ratio + density_score + six_score - comp_penalty

    # ------------------------------------------------------------------
    # For a chosen subset of panels, search all symmetry combinations.
    # ------------------------------------------------------------------
    def best_for_panels(panels):
        num_panels = len(panels)
        num_syms = len(panels[0])
        best_score = -1e9
        best_out = None

        target_h, target_w = panels[0][0].shape

        # General iterative product with small depth (usually 4).
        def rec(idx, cur):
            nonlocal best_score, best_out
            if idx == num_panels:
                out = color_map[cur]
                s = score_output(out, has_six_input)
                if s > best_score:
                    best_score = s
                    best_out = out.copy()
                return

            for sym_idx in range(num_syms):
                arr = panels[idx][sym_idx]
                # All panels should have the same shape; if not, pad/skip.
                if arr.shape != (target_h, target_w):
                    tmp = np.zeros((target_h, target_w), dtype=np.uint32)
                    tmp[:arr.shape[0], :arr.shape[1]] = arr
                    arr = tmp
                rec(idx + 1, cur ^ arr)

        rec(0, np.zeros((target_h, target_w), dtype=np.uint32))
        return best_out, best_score

    # ------------------------------------------------------------------
    # Determine whether any candidate panel contains color 6.
    # ------------------------------------------------------------------
    has_six_input = False
    for cand in pool:
        r0, c0, r1, c1 = cand["rect"]
        if np.any(grid[r0 + 1:r1, c0 + 1:c1] == 6):
            has_six_input = True
            break

    # ------------------------------------------------------------------
    # Choose the best subset + symmetry combination.
    # Prefer 4 panels when available; otherwise use the available ones.
    # ------------------------------------------------------------------
    best_global = None
    best_global_score = -1e9

    n_pool = len(pool)

    # Candidate subset sizes to try. Prefer 4, then 3, then 2/1.
    subset_sizes = [4, 3, 2, 1]
    subset_sizes = [s for s in subset_sizes if s <= n_pool]
    if not subset_sizes:
        return np.zeros_like(grid)

    for size in subset_sizes:
        # For size 4, if many candidates exist, only inspect a reasonable pool.
        if size == 4 and n_pool > 6:
            indices_pool = range(6)
        else:
            indices_pool = range(n_pool)

        for idxs in combinations(indices_pool, size):
            panels = prepare_panels(idxs)
            if not panels:
                continue
            out, s = best_for_panels(panels)
            if out is None:
                continue
            # Small bonus for using four panels, since the examples do.
            s += 0.05 * size
            if s > best_global_score:
                best_global_score = s
                best_global = out

    if best_global is None:
        return np.zeros_like(grid)

    return best_global.astype(grid.dtype, copy=False)