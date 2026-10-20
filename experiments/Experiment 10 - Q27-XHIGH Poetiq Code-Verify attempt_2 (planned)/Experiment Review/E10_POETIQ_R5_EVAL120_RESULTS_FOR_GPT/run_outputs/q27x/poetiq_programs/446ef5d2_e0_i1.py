import numpy as np
from collections import Counter
from itertools import product

def _prefix_sum(mask):
    """2D prefix sum for an integer mask."""
    pref = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=int)
    pref[1:, 1:] = mask.astype(int).cumsum(axis=0).cumsum(axis=1)
    return pref

def _rect_sum(pref, r1, c1, r2, c2):
    """Sum on inclusive rectangle [r1:r2+1, c1:c2+1]."""
    return pref[r2 + 1, c2 + 1] - pref[r1, c2 + 1] - pref[r2 + 1, c1] + pref[r1, c1]

def _side_all_color(grid, r1, c1, r2, c2, color, side):
    if side == "top":
        return int(np.all(grid[r1, c1:c2 + 1] == color))
    if side == "bottom":
        return int(np.all(grid[r2, c1:c2 + 1] == color))
    if side == "left":
        return int(np.all(grid[r1:r2 + 1, c1] == color))
    if side == "right":
        return int(np.all(grid[r1:r2 + 1, c2] == color))
    return False

def _candidate_score(grid, r1, c1, r2, c2, border):
    """
    Prefer larger solid tiles that contain non-border payload and have
    fewer background holes.
    """
    block = grid[r1:r2 + 1, c1:c2 + 1]
    area = block.size
    non_border = int(np.sum(block != border))
    non_bg = int(np.sum(block != grid.flat[np.argmax(np.bincount(grid.flat))]))
    bg = int(np.sum(block == grid.flat[np.argmax(np.bincount(grid.flat))]))
    return area * 100 + non_border * 10 - bg

def _find_four_corner_solution(grid, bg, border, allow_background_inside):
    """
    Try to find four rectangular pieces:
      TL: top + left border
      TR: top + right border
      BL: bottom + left border
      BR: bottom + right border

    The pieces must be compatible:
      TL/TR same height, BL/BR same height,
      TL/BL same width, TR/BR same width.
    """
    H, W = grid.shape
    border_mask = (grid == border).astype(int)
    nonbg_mask = (grid != bg).astype(int)
    pref_border = _prefix_sum(border_mask)
    pref_nonbg = _prefix_sum(nonbg_mask)
    total_nonbg = int(pref_nonbg[H, W])

    corners = {"TL": [], "TR": [], "BL": [], "BR": []}

    # Enumerate all rectangles.  ARC grids in this task are small.
    for r1 in range(H - 1):
        for r2 in range(r1 + 1, H):
            h = r2 - r1 + 1
            for c1 in range(W - 1):
                for c2 in range(c1 + 1, W):
                    w = c2 - c1 + 1

                    if not allow_background_inside:
                        if _rect_sum(_prefix_sum((grid == bg).astype(int)), r1, c1, r2, c2) > 0:
                            pass  # recompute below once for all? keep simple
                    bg_inside = _rect_sum(_prefix_sum((grid == bg).astype(int)), r1, c1, r2, c2)
                    if not allow_background_inside and bg_inside > 0:
                        continue

                    # Must contain some non-border payload; pure border rectangles are not tiles.
                    border_inside = _rect_sum(pref_border, r1, c1, r2, c2)
                    if border_inside == h * w:
                        continue

                    top = _side_all_color(grid, r1, c1, r2, c2, border, "top")
                    bottom = _side_all_color(grid, r1, c1, r2, c2, border, "bottom")
                    left = _side_all_color(grid, r1, c1, r2, c2, border, "left")
                    right = _side_all_color(grid, r1, c1, r2, c2, border, "right")

                    rec = (r1, c1, r2, c2, h, w)
                    score = _candidate_score(grid, r1, c1, r2, c2, border)

                    if top and left:
                        corners["TL"].append((score, rec))
                    if top and right:
                        corners["TR"].append((score, rec))
                    if bottom and left:
                        corners["BL"].append((score, rec))
                    if bottom and right:
                        corners["BR"].append((score, rec))

    # Keep only a reasonable number of best candidates per (h,w) to avoid blowup.
    for name in corners:
        by_dim = {}
        for score, rec in corners[name]:
            h, w = rec[4], rec[5]
            by_dim.setdefault((h, w), []).append((score, rec))
        cleaned = []
        for dim, lst in by_dim.items():
            lst.sort(reverse=True)
            cleaned.extend(lst[:20])
        cleaned.sort(reverse=True)
        corners[name] = cleaned[:80]

    if not all(corners[k] for k in corners):
        return None

    def overlap(a, b):
        ar1, ac1, ar2, ac2 = a[0], a[1], a[2], a[3]
        br1, bc1, br2, bc2 = b[0], b[1], b[2], b[3]
        return not (ar2 < br1 or br2 < ar1 or ac2 < bc1 or bc2 < ac1)

    best = None

    # Try all dimension tuples that are feasible.
    tl_dims = {(r[4], r[5]) for _, r in corners["TL"]}
    tr_dims = {(r[4], r[5]) for _, r in corners["TR"]}
    bl_dims = {(r[4], r[5]) for _, r in corners["BL"]}
    br_dims = {(r[4], r[5]) for _, r in corners["BR"]}

    for (ht, wt_left), (ht2, wt_right) in product(tl_dims, tr_dims):
        if ht != ht2:
            continue
        for (hb, wb_left), (hb2, wb_right) in product(bl_dims, br_dims):
            if hb != hb2:
                continue
            if wt_left != wb_left:
                continue
            if wt_right != wb_right:
                continue

            tl_list = [r for _, r in corners["TL"] if r[4] == ht and r[5] == wt_left]
            tr_list = [r for _, r in corners["TR"] if r[4] == ht and r[5] == wt_right]
            bl_list = [r for _, r in corners["BL"] if r[4] == hb and r[5] == wb_left]
            br_list = [r for _, r in corners["BR"] if r[4] == hb and r[5] == wb_right]

            if not (tl_list and tr_list and bl_list and br_list):
                continue

            # Limit nested search.
            tl_list = tl_list[:25]
            tr_list = tr_list[:25]
            bl_list = bl_list[:25]
            br_list = br_list[:25]

            for tl in tl_list:
                if overlap(tl, tl):
                    continue
                for tr in tr_list:
                    if overlap(tl, tr):
                        continue
                    for bl in bl_list:
                        if overlap(tl, bl) or overlap(tr, bl):
                            continue
                        for br in br_list:
                            if overlap(tl, br) or overlap(tr, br) or overlap(bl, br):
                                continue

                            covered = (
                                _rect_sum(pref_nonbg, tl[0], tl[1], tl[2], tl[3]) +
                                _rect_sum(pref_nonbg, tr[0], tr[1], tr[2], tr[3]) +
                                _rect_sum(pref_nonbg, bl[0], bl[1], bl[2], bl[3]) +
                                _rect_sum(pref_nonbg, br[0], br[1], br[2], br[3])
                            )

                            # Prefer solutions that explain most non-background pixels.
                            score = covered * 1000 + (ht + hb + wt_left + wt_right) * 10
                            if best is None or score > best[0]:
                                best = (score, {"TL": tl, "TR": tr, "BL": bl, "BR": br})

    return best

def _assemble_four_corner(grid, bg, border, solution):
    """Assemble the four tiles into a 2x2 composite grid."""
    score, tiles = solution
    tl, tr, bl, br = tiles["TL"], tiles["TR"], tiles["BL"], tiles["BR"]

    ht, wl = tl[4], tl[5]
    hr, wr = tr[4], tr[5]
    hl, ll = bl[4], bl[5]
    hb, rb = br[4], br[5]

    # Compatibility should have been checked, but be defensive.
    if ht != hr:
        ht = max(ht, hr)
    if hl != hb:
        hl = max(hl, hb)
    if wl != ll:
        wl = max(wl, ll)
    if wr != rb:
        wr = max(wr, rb)

    H = ht + hl
    W = wl + wr
    out = np.full((H, W), bg, dtype=grid.dtype)

    def place(tile, r0, c0, max_h, max_w):
        r1, c1, r2, c2 = tile[0], tile[1], tile[2], tile[3]
        block = grid[r1:r2 + 1, c1:c2 + 1]
        h, w = block.shape
        out[r0:r0 + h, c0:c0 + w] = block

    place(tl, 0, 0, ht, wl)
    place(tr, 0, wl, ht, wr)
    place(bl, ht, 0, hl, wl)
    place(br, ht, wl, hl, wr)

    # Enforce the outer border of the final object.
    out[0, 0:wl + wr] = border
    out[-1, 0:wl + wr] = border
    out[0:ht + hl, 0] = border
    out[0:ht + hl, -1] = border

    return out

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a 2x2 composite image from scattered corner pieces.

    The rule:
      * The most common color is background.
      * One non-background color acts as the border/frame color.
      * Four rectangular pieces exist in the input.  Each piece has the
        frame color on the sides that point to its final position:
            top-left piece has top + left sides bordered,
            top-right piece has top + right sides bordered,
            bottom-left piece has bottom + left sides bordered,
            bottom-right piece has bottom + right sides bordered.
      * The pieces are placed into a 2x2 grid according to those sides.

    The implementation searches over possible border colors and valid
    rectangles, chooses a compatible set of four non-overlapping pieces that
    covers the most non-background pixels, and assembles them.
    """
    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    bg = int(np.bincount(grid.flatten()).argmax())

    # Try likely border colors first, but allow fallback.
    color_counts = Counter(int(x) for x in grid.flatten())
    candidates = [c for c, n in color_counts.items() if c != bg and n > 0]
    candidates.sort(key=lambda c: (-color_counts[c], c))

    best_solution = None
    best_border = None
    best_global_score = -1

    for border in candidates:
        for allow_bg_inside in (False, True):
            sol = _find_four_corner_solution(grid, bg, border, allow_bg_inside)
            if sol is None:
                continue
            # A penalty if background holes are allowed.
            score = sol[0] - (10 if allow_bg_inside else 0)
            if score > best_global_score:
                best_global_score = score
                best_solution = sol
                best_border = border

    if best_solution is not None:
        return _assemble_four_corner(grid, bg, best_border, best_solution)

    # Fallback: crop the bounding box of all non-background pixels.
    nonbg = (grid != bg)
    if not np.any(nonbg):
        return np.zeros_like(grid)

    rows = np.any(nonbg, axis=1)
    cols = np.any(nonbg, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return grid[rmin:rmax + 1, cmin:cmax + 1].copy()