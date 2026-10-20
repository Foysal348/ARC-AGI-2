import numpy as np
from collections import Counter


class _SearchLimit(Exception):
    pass


def _build_prefix(mask):
    """2D prefix sum for a 0/1 mask, stored as nested Python lists."""
    h, w = mask.shape
    pref = [[0] * (w + 1) for _ in range(h + 1)]
    for i in range(h):
        row_sum = 0
        pref_i = pref[i]
        pref_next = pref[i + 1]
        for j in range(w):
            row_sum += mask[i, j]
            pref_next[j + 1] = pref_i[j + 1] + row_sum
    return pref


def _rect_sum(pref, r1, c1, r2, c2):
    """Inclusive rectangle sum from prefix."""
    return (pref[r2 + 1][c2 + 1]
            - pref[r1][c2 + 1]
            - pref[r2 + 1][c1]
            + pref[r1][c1])


class _Piece:
    __slots__ = (
        "id", "r1", "c1", "r2", "c2", "h", "w", "area",
        "input_r", "input_c", "grid", "T", "B", "L", "R", "sides"
    )

    def __init__(self, pid, r1, c1, r2, c2, grid, T, B, L, R):
        self.id = pid
        self.r1, self.c1, self.r2, self.c2 = r1, c1, r2, c2
        self.h = r2 - r1 + 1
        self.w = c2 - c1 + 1
        self.area = self.h * self.w
        self.input_r = r1
        self.input_c = c1
        self.grid = grid[r1:r2 + 1, c1:c2 + 1].copy()
        self.T = bool(T)
        self.B = bool(B)
        self.L = bool(L)
        self.R = bool(R)
        self.sides = int(T) + int(B) + int(L) + int(R)


def _extract_pieces(grid, bg, border):
    """
    Extract maximal rectangular puzzle pieces.

    A candidate piece is a solid rectangle (no background cells) whose at
    least two full sides are made of the border color.  It is kept only if
    it is not contained in a larger valid rectangle.  Overlapping maximal
    candidates are resolved greedily by area.
    """
    H, W = grid.shape

    bg_mask = (grid == bg).astype(int)
    border_mask = (grid == border).astype(int)

    bg_pref = _build_prefix(bg_mask)
    border_pref = _build_prefix(border_mask)

    # Row/column prefix sums for checking whether an entire side is border color.
    row_border_pref = [[0] * (W + 1) for _ in range(H)]
    for r in range(H):
        acc = 0
        row = row_border_pref[r]
        for c in range(W):
            acc += border_mask[r, c]
            row[c + 1] = acc

    col_border_pref = [[0] * (H + 1) for _ in range(W)]
    for c in range(W):
        acc = 0
        col = col_border_pref[c]
        for r in range(H):
            acc += border_mask[r, c]
            col[r + 1] = acc

    candidates = set()

    # Enumerate all solid rectangles with height/width >= 2.
    for r1 in range(H):
        for r2 in range(r1 + 1, H):
            h = r2 - r1 + 1

            # Columns that are solid over rows r1..r2.
            solid_cols = [False] * W
            for c in range(W):
                if _rect_sum(bg_pref, r1, c, r2, c) == 0:
                    solid_cols[c] = True

            c = 0
            while c < W:
                if not solid_cols[c]:
                    c += 1
                    continue

                c1 = c
                while c < W and solid_cols[c]:
                    c += 1
                c_end = c - 1

                for c2 in range(c1 + 1, c_end + 1):
                    w = c2 - c1 + 1

                    top = row_border_pref[r1][c2 + 1] - row_border_pref[r1][c1] == w
                    bottom = row_border_pref[r2][c2 + 1] - row_border_pref[r2][c1] == w
                    left = col_border_pref[c1][r2 + 1] - col_border_pref[c1][r1] == h
                    right = col_border_pref[c2][r2 + 1] - col_border_pref[c2][r1] == h

                    sides = int(top) + int(bottom) + int(left) + int(right)
                    if sides < 2:
                        continue

                    area = h * w
                    border_count = _rect_sum(border_pref, r1, c1, r2, c2)

                    # Ignore tiny all-border noise blocks.  Larger all-border
                    # rectangles are allowed because they may be real filler pieces.
                    if border_count == area and area < 9:
                        continue

                    candidates.add((r1, c1, r2, c2, area, sides, top, bottom, left, right))

    if not candidates:
        return []

    # Keep only inclusion-maximal valid rectangles.
    candidates = sorted(candidates, key=lambda x: (x[4], x[5]), reverse=True)

    # Defensive cap for pathological cases.
    if len(candidates) > 20000:
        candidates = candidates[:20000]

    maximal = []
    for cand in candidates:
        r1, c1, r2, c2 = cand[0], cand[1], cand[2], cand[3]
        contained = False
        for m in maximal:
            if m[0] <= r1 and m[1] <= c1 and m[2] >= r2 and m[3] >= c2:
                contained = True
                break
        if not contained:
            maximal.append(cand)
            if len(maximal) > 500:
                break

    # If maximal rectangles overlap, keep a non-overlapping subset by area.
    maximal.sort(key=lambda x: x[4], reverse=True)
    selected = []
    for cand in maximal:
        r1, c1, r2, c2 = cand[0], cand[1], cand[2], cand[3]
        overlap = False
        for p in selected:
            pr1, pc1, pr2, pc2 = p[0], p[1], p[2], p[3]
            if not (r2 < pr1 or pr2 < r1 or c2 < pc1 or pc2 < c1):
                overlap = True
                break
        if not overlap:
            selected.append(cand)

    pieces = []
    for i, cand in enumerate(selected):
        r1, c1, r2, c2, area, sides, top, bottom, left, right = cand
        pieces.append(_Piece(i, r1, c1, r2, c2, grid, top, bottom, left, right))

    return pieces


def _subset_sum_possible(lengths, target):
    """Necessary subset-sum test for tiling an outer edge."""
    if target == 0:
        return True
    dp = [False] * (target + 1)
    dp[0] = True
    for length in lengths:
        if length <= target:
            for s in range(target, length - 1, -1):
                if dp[s - length]:
                    dp[s] = True
    return dp[target]


def _edge_feasible(pieces, H, W):
    """Quick necessary conditions for the outer border to be possible."""
    if not any(p.T and p.L for p in pieces):
        return False
    if not any(p.T and p.R for p in pieces):
        return False
    if not any(p.B and p.L for p in pieces):
        return False
    if not any(p.B and p.R for p in pieces):
        return False

    if not _subset_sum_possible([p.w for p in pieces if p.T], W):
        return False
    if not _subset_sum_possible([p.w for p in pieces if p.B], W):
        return False
    if not _subset_sum_possible([p.h for p in pieces if p.L], H):
        return False
    if not _subset_sum_possible([p.h for p in pieces if p.R], H):
        return False

    return True


def _solve_packing(pieces, H, W, node_limit=300000):
    """
    Exact rectangle-packing search for one candidate final size H x W.

    The first empty cell in row-major order must be the top-left corner of
    the next piece, so this gives a complete backtracking search over all
    non-rotated tilings.
    """
    N = len(pieces)
    if N == 0:
        return None

    full_empty = (1 << (H * W)) - 1
    all_placed = (1 << N) - 1

    # Precompute bit masks and edge legality for each piece at each top-left.
    masks = [dict() for _ in range(N)]
    edge_ok = [dict() for _ in range(N)]

    for pi, p in enumerate(pieces):
        row_mask = (1 << p.w) - 1
        for r in range(H - p.h + 1):
            for c in range(W - p.w + 1):
                mask = 0
                for dr in range(p.h):
                    mask |= row_mask << ((r + dr) * W + c)

                ok = True
                if r == 0 and not p.T:
                    ok = False
                if r + p.h == H and not p.B:
                    ok = False
                if c == 0 and not p.L:
                    ok = False
                if c + p.w == W and not p.R:
                    ok = False

                if ok:
                    masks[pi][(r, c)] = mask
                    edge_ok[pi][(r, c)] = True

    # Candidate lists per first-empty cell.
    cell_candidates = [[None for _ in range(W)] for __ in range(H)]
    for r in range(H):
        for c in range(W):
            lst = []
            for pi in range(N):
                if (r, c) in edge_ok[pi]:
                    lst.append(pi)
            lst.sort(key=lambda i: (-pieces[i].area, -pieces[i].sides, i))
            cell_candidates[r][c] = lst

    cache = {}
    nodes = 0

    def dfs(empty_mask, placed_mask):
        nonlocal nodes
        key = (empty_mask, placed_mask)
        if key in cache:
            return cache[key]

        nodes += 1
        if nodes > node_limit:
            raise _SearchLimit()

        if placed_mask == all_placed:
            cache[key] = ()
            return ()

        if empty_mask == 0:
            cache[key] = None
            return None

        lsb = empty_mask & -empty_mask
        idx = lsb.bit_length() - 1
        r = idx // W
        c = idx % W

        for pi in cell_candidates[r][c]:
            if placed_mask & (1 << pi):
                continue

            mask = masks[pi].get((r, c), 0)
            if not mask:
                continue
            if (empty_mask & mask) != mask:
                continue

            res = dfs(empty_mask ^ mask, placed_mask | (1 << pi))
            if res is not None:
                ans = ((pi, r, c),) + res
                cache[key] = ans
                return ans

        cache[key] = None
        return None

    try:
        return dfs(full_empty, 0)
    except _SearchLimit:
        return None


def _candidate_dimensions(pieces, grid_h, grid_w):
    total_area = sum(p.area for p in pieces)
    if total_area == 0:
        return []

    max_h = max(p.h for p in pieces)
    max_w = max(p.w for p in pieces)

    divs = []
    for h in range(1, grid_h + 1):
        if total_area % h == 0:
            w = total_area // h
            if w <= grid_w and h >= max_h and w >= max_w:
                divs.append((h, w))

    if not divs:
        return []

    # Prefer dimensions close to the scattered pieces' input bounding box.
    min_r = min(p.input_r for p in pieces)
    max_r = max(p.r2 for p in pieces)
    min_c = min(p.input_c for p in pieces)
    max_c = max(p.c2 for p in pieces)
    bh = max_r - min_r + 1
    bw = max_c - min_c + 1

    def score(d):
        h, w = d
        if bh == 0 or bw == 0 or w == 0:
            return (0.0, h + w, h, w)
        aspect_diff = abs((h / w) - (bh / bw))
        size_diff = abs(h - bh) + abs(w - bw)
        return (aspect_diff, size_diff, h, w)

    divs.sort(key=score)
    return divs


def _choose_offset(H, W, placements, pieces, grid_h, grid_w):
    """
    Choose the translation of the assembled rectangle.

    The examples keep at least one piece in its original absolute position.
    Therefore we first consider translations that make at least one piece's
    top-left corner unchanged, and choose the one minimizing the maximum
    Manhattan movement of any piece.  If no such translation exists, fall
    back to the best translation over all valid placements.
    """
    if H > grid_h or W > grid_w:
        return None

    offsets = []

    # Translations that keep one piece exactly where it is.
    for pi, r, c in placements:
        p = pieces[pi]
        R0 = p.input_r - r
        C0 = p.input_c - c
        if 0 <= R0 <= grid_h - H and 0 <= C0 <= grid_w - W:
            offsets.append((R0, C0))

    # Fallback: any valid position of the final rectangle.
    if not offsets:
        for R0 in range(grid_h - H + 1):
            for C0 in range(grid_w - W + 1):
                offsets.append((R0, C0))

    best = None
    for R0, C0 in offsets:
        max_move = 0
        sum_move = 0
        fixed_area = 0

        for pi, r, c in placements:
            p = pieces[pi]
            dr = (R0 + r) - p.input_r
            dc = (C0 + c) - p.input_c
            mv = abs(dr) + abs(dc)
            max_move = max(max_move, mv)
            sum_move += mv
            if mv == 0:
                fixed_area += p.area

        key = (max_move, sum_move, -fixed_area, R0, C0)
        if best is None or key < best[0]:
            best = (key, (R0, C0))

    return best[1] if best is not None else None


def _evaluate_and_assemble(grid, bg, pieces, placements, H, W, offset):
    """Assemble output and compute a global score for comparing candidates."""
    H_in, W_in = grid.shape
    out = np.full((H_in, W_in), bg, dtype=grid.dtype)

    R0, C0 = offset
    max_move = 0
    sum_move = 0
    fixed_area = 0

    for pi, r, c in placements:
        p = pieces[pi]
        rr = R0 + r
        cc = C0 + c
        out[rr:rr + p.h, cc:cc + p.w] = p.grid

        dr = rr - p.input_r
        dc = cc - p.input_c
        mv = abs(dr) + abs(dc)
        max_move = max(max_move, mv)
        sum_move += mv
        if mv == 0:
            fixed_area += p.area

    non_bg = int(np.sum(grid != bg))
    total_area = sum(p.area for p in pieces)
    coverage = total_area / max(1, non_bg)

    # Higher coverage first, then smaller movement.
    score = (-coverage, max_move, sum_move, -fixed_area, H, W)
    return score, out


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a rectangular picture from scattered rectangular pieces.

    The input contains rectangular puzzle pieces on a background color.
    Each piece has one or more full sides in the frame/border color.  The
    output is the assembled rectangle, placed in the original canvas.

    The algorithm:
      1. Guess the background color (most frequent color).
      2. For plausible border colors, extract maximal solid rectangles that
         have at least two full sides in the border color.
      3. Pack those rectangles into the smallest rectangle(s) possible
         while enforcing that every outer edge of the final rectangle is
         covered by pieces whose corresponding side is entirely the border
         color.
      4. Choose the translation that keeps at least one piece in its original
         position if possible, minimizing the maximum movement of pieces.
    """
    grid = np.asarray(grid)
    if grid.size == 0 or grid.ndim != 2:
        return grid.copy()

    H_in, W_in = grid.shape
    if H_in < 2 or W_in < 2:
        return grid.copy()

    counts = Counter(int(x) for x in grid.flat)
    if not counts:
        return grid.copy()

    bg = int(counts.most_common(1)[0][0])
    non_bg_colors = [c for c, _ in counts.most_common() if c != bg]

    best = None

    # Try the most frequent non-background colors as border color.
    for border in non_bg_colors[:6]:
        pieces = _extract_pieces(grid, bg, border)
        if not pieces:
            continue

        total_area = sum(p.area for p in pieces)
        if total_area == 0:
            continue

        dims = _candidate_dimensions(pieces, H_in, W_in)
        if not dims:
            continue

        for H, W in dims:
            if not _edge_feasible(pieces, H, W):
                continue

            placements = _solve_packing(pieces, H, W, node_limit=300000)
            if placements is None:
                continue

            offset = _choose_offset(H, W, placements, pieces, H_in, W_in)
            if offset is None:
                continue

            score, assembled = _evaluate_and_assemble(grid, bg, pieces, placements, H, W, offset)

            if best is None or score < best[0]:
                best = (score, assembled)

            # If this solution is very good, stop early.
            # The first term is -coverage; coverage is a float in [0, 1].
            if best[0][0] > -0.999999 and best[0][1] <= 12:
                break

    if best is not None:
        return best[1]

    # Fallback: center the bounding box of non-background pixels.
    non_bg = (grid != bg)
    if not np.any(non_bg):
        return np.zeros_like(grid)

    rows = np.any(non_bg, axis=1)
    cols = np.any(non_bg, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    crop = grid[rmin:rmax + 1, cmin:cmax + 1]
    h = crop.shape[0]
    w = crop.shape[1]

    out = np.full((H_in, W_in), bg, dtype=grid.dtype)
    r0 = max(0, (H_in - h) // 2)
    c0 = max(0, (W_in - w) // 2)
    out[r0:r0 + h, c0:c0 + w] = crop
    return out