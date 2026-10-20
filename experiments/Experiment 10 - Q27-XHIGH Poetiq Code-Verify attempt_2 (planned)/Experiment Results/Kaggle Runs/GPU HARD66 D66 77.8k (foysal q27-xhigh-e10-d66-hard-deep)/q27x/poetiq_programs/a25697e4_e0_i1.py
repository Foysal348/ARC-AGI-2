import numpy as np
from collections import Counter
from itertools import product

def _components(mask):
    """Return 4-connected components as a list of (pixels, bbox)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                stack = [(i, j)]
                seen[i, j] = True
                pix = []
                while stack:
                    r, c = stack.pop()
                    pix.append((r, c))
                    for nr, nc in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                r0 = min(p[0] for p in pix)
                r1 = max(p[0] for p in pix)
                c0 = min(p[1] for p in pix)
                c1 = max(p[1] for p in pix)
                comps.append((pix, (r0, r1, c0, c1)))
    return comps

def _crop_mask(grid, color, bbox=None):
    h, w = grid.shape
    if bbox is None:
        ys, xs = np.where(grid == color)
        if len(ys) == 0:
            return np.zeros((0, 0), dtype=int)
        r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
        bbox = (r0, r1, c0, c1)
    r0, r1, c0, c1 = bbox
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in zip(*np.where(grid[r0:r1+1, c0:c1+1] == color)):
        sub[r, c] = 1
    return sub

def _all_orientations(mask):
    if mask.size == 0:
        return [mask]
    out = []
    def add(m):
        m = np.asarray(m)
        if 0 not in [m.shape[0], m.shape[1]] and m.sum() > 0:
            for prev in out:
                if prev.shape == m.shape and np.array_equal(prev, m):
                    return
            out.append(m)
    cur = mask
    for _ in range(4):
        add(cur)
        add(np.fliplr(cur))
        cur = np.rot90(cur)
    return out

def _place_mask(canvas, mask, r, c, color):
    h, w = canvas.shape
    mh, mw = mask.shape
    if r < 0 or c < 0 or r + mh > h or c + mw > w:
        return False
    sub = canvas[r:r+mh, c:c+mw]
    if np.any((sub != 0) & (mask != 0)):
        return False
    canvas[r:r+mh, c:c+mw] = np.where(mask != 0, color, sub)
    return True

def _bbox_area(mask):
    ys, xs = np.where(mask != 0)
    if len(ys) == 0:
        return 0
    return (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by identifying the dominant anchor object and compacting
    the remaining objects around it.

    The rule used:
      1. The background is the most frequent color.
      2. The anchor is the non-background color with the best compactness
         (high pixel count and small bounding-box area).
      3. The anchor is kept in place.
      4. Every other connected component is moved, optionally rotated/reflected,
         to the best location that:
           - minimizes overlap with already placed pixels,
           - minimizes the total bounding-box area of all non-background pixels,
           - stays close to the anchor.
      5. Colors are preserved in the output.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, dtype=int, copy=True)
    h, w = grid.shape

    # Background color: most common value.
    bg = Counter(grid.flatten()).most_common(1)[0][0]

    # Non-background components.
    nb_mask = grid != bg
    if not nb_mask.any():
        return grid

    comp_list = _components(nb_mask)
    if not comp_list:
        return grid

    # Score colors: prefer high pixel count and compact bounding boxes.
    color_count = Counter()
    color_area = Counter()
    for pix, (r0, r1, c0, c1) in comp_list:
        col = grid[pix[0][0], pix[0][1]]
        color_count[col] += len(pix)
        color_area[col] += (r1 - r0 + 1) * (c1 - c0 + 1)

    def color_score(col):
        cnt = color_count[col]
        area = max(1, color_area[col])
        # Compactness bonus: dense components are preferred as anchor.
        return (cnt, -area / max(1, cnt))

    non_bg_colors = [c for c in color_count if c != bg]
    if not non_bg_colors:
        return grid

    anchor_color = max(non_bg_colors, key=color_score)

    out = np.full((h, w), bg, dtype=int)

    # Place all anchor pixels exactly where they are.
    for r in range(h):
        for c in range(w):
            if grid[r, c] == anchor_color:
                out[r, c] = anchor_color

    # Other components to place.
    others = []
    for pix, bbox in comp_list:
        col = grid[pix[0][0], pix[0][1]]
        if col == anchor_color:
            continue
        r0, r1, c0, c1 = bbox
        mask = _crop_mask(grid, col, bbox)
        others.append({
            "color": col,
            "mask": mask,
            "bbox": bbox,
            "center": (
                sum(p[0] for p in pix) / max(1, len(pix)),
                sum(p[1] for p in pix) / max(1, len(pix))
            )
        })

    # Anchor center for proximity scoring.
    ar, ac = np.where(grid == anchor_color)
    if len(ar) > 0:
        anchor_center = (ar.mean(), ac.mean())
    else:
        anchor_center = (h / 2.0, w / 2.0)

    # Greedy placement of remaining components.
    # For each component, test orientations and nearby translations.
    placed = []
    others.sort(key=lambda x: -x["mask"].size)

    for obj in others:
        base_mask = obj["mask"]
        color = obj["color"]
        r0, r1, c0, c1 = obj["bbox"]

        best = None
        best_score = None

        # Candidate orientations.
        orientations = _all_orientations(base_mask)

        # Candidate translation ranges: original position plus nearby shifts.
        max_shift = 12
        row_range = range(max(0, r0 - max_shift), min(h, r1 + max_shift + 1))
        col_range = range(max(0, c0 - max_shift), min(w, c1 + max_shift + 1))

        for orient in orientations:
            oh, ow = orient.shape
            for rr in range(max(0, r0 - max_shift), min(h - oh + 1, r1 + max_shift + 1)):
                for cc in range(max(0, c0 - max_shift), min(w - ow + 1, c1 + max_shift + 1)):
                    # Temporary placement check.
                    test = out.copy()
                    if not _place_mask(test, orient, rr, cc, color):
                        continue

                    # Compute score.
                    nb = test != bg
                    ys, xs = np.where(nb)
                    if len(ys) == 0:
                        area = 0
                    else:
                        area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)

                    # Proximity to anchor center.
                    pr = rr + oh / 2.0
                    pc = cc + ow / 2.0
                    prox = abs(pr - anchor_center[0]) + abs(pc - anchor_center[1])

                    # Keep near original location to avoid wild moves.
                    orig_r = (r0 + r1) / 2.0
                    orig_c = (c0 + c1) / 2.0
                    dist = abs(pr - orig_r) + abs(pc - orig_c)

                    # Prefer placements that do not create huge new bounding boxes.
                    score = (area, int(prox * 10), int(dist * 10))

                    if best_score is None or score < best_score:
                        best_score = score
                        best = (orient, rr, cc)

        if best is not None:
            orient, rr, cc = best
            _place_mask(out, orient, rr, cc, color)
            placed.append(color)
        else:
            # Fallback: keep original component in place if no non-overlap spot found.
            for r, c in [(r0 + x, c0 + y) for x, y in [(0, 0)]]:
                _place_mask(out, base_mask, r, c, color)

    return out