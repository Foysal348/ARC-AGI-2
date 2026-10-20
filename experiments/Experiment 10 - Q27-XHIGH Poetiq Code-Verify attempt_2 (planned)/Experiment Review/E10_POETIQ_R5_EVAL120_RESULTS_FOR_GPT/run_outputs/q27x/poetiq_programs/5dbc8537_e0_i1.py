import numpy as np
from collections import deque, Counter
from itertools import product

def _connected_components(mask):
    """Return 4-connected components of True cells as lists of coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                comp = []
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                comps.append(comp)
    return comps

def _crop_pattern(grid, comp):
    """Crop the minimal bounding box for a component."""
    rows = [r for r, c in comp]
    cols = [c for r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return grid[r0:r1 + 1, c0:c1 + 1], r0, c0, r1 - r0 + 1, c1 - c0 + 1

def _rectangles_in_mask(mask, max_rectangles=2000):
    """
    Enumerate all-True axis-aligned rectangles in a binary mask.
    To keep search manageable, only maximal-ish candidates are used:
    for each top row and left column, extend as far right/down as possible,
    and also record several smaller suffixes.
    """
    h, w = mask.shape
    rects = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0]:
                continue
            # maximum rectangle with top-left (r0,c0)
            max_c1 = c0
            max_r1 = r0
            while max_c1 + 1 < w and mask[r0, max_c1 + 1]:
                max_c1 += 1
            for r1 in range(r0, h):
                if not all(mask[r1, c] for c in range(c0, max_c1 + 1)):
                    break
                # add a few widths, especially full and small ones
                for w_len in {1, 2, 3, 4, 5, max_c1 - c0 + 1}:
                    c1 = min(max_c1, c0 + w_len - 1)
                    if all(mask[r1, c] for c in range(c0, c1 + 1)):
                        rects.append((r0, c0, r1, c1))
            if len(rects) > max_rectangles:
                break
        if len(rects) > max_rectangles:
            break
    # deduplicate
    return list(set(rects))

def _resize_pattern(pattern, h, w):
    """Nearest-neighbor resize a pattern to (h,w)."""
    if pattern.shape == (h, w):
        return pattern.copy()
    ph, pw = pattern.shape
    res = np.zeros((h, w), dtype=pattern.dtype)
    for y in range(h):
        for x in range(w):
            yy = min(ph - 1, int(y * ph / h))
            xx = min(pw - 1, int(x * pw / w))
            res[y, x] = pattern[yy, xx]
    return res

def _find_template_component(grid):
    """
    Find a connected component whose bounding box contains only:
      - its own color
      - exactly one other "hole" color
    This is the intended empty frame/canvas in the examples.
    """
    h, w = grid.shape
    colors = set(grid.ravel().tolist())
    best = None  # (score, color, comp, hole_color)

    # For each color, get components and test bbox purity.
    for color in colors:
        mask = (grid == color)
        for comp in _connected_components(mask):
            rows = [r for r, c in comp]
            cols = [c for r, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            box = grid[r0:r1 + 1, c0:c1 + 1]
            box_colors = set(box.ravel().tolist())
            if len(box_colors) != 2:
                continue
            if color not in box_colors:
                continue
            hole_color = (box_colors - {color}).pop()

            # Require at least some holes inside the component's bbox.
            hole_count = int(np.sum(box == hole_color))
            comp_size = len(comp)
            if hole_count <= 0:
                continue

            # Prefer larger, more "frame-like" components.
            # Penalize components that are tiny or solid.
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            score = hole_count * 10 + comp_size - 0.1 * area

            if best is None or score > best[0]:
                best = (score, color, comp, hole_color, r0, r1, c0, c1)

    return best

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform by extracting a framed template and filling its empty holes
    with colored objects from outside the template.

    The algorithm:
    1. Find a connected component that looks like a frame/canvas:
       inside its bounding box there are only the component color and one
       hole color.
    2. Use that bounding box as the output canvas, keeping the component color.
    3. Extract objects from outside the canvas (connected components of colors
       other than the canvas color and the hole color).
    4. Place the objects into the hole mask by exact-rectangle packing when
       possible, allowing nearest-neighbor resizing for close mismatches.
    """
    grid = np.array(grid, dtype=int)
    if grid.size == 0:
        return grid.copy()

    template = _find_template_component(grid)

    # Fallback: if no clean template is found, return the input unchanged.
    if template is None:
        return grid.copy()

    _, frame_color, frame_comp, hole_color, r0, r1, c0, c1 = template
    H, W = r1 - r0 + 1, c1 - c0 + 1

    out = np.full((H, W), frame_color, dtype=int)
    hole_mask = np.zeros((H, W), dtype=bool)
    for dr in range(H):
        for dc in range(W):
            if grid[r0 + dr, c0 + dc] != frame_color:
                hole_mask[dr, dc] = True

    # Extract outside pieces.
    outside_mask = np.ones_like(grid, dtype=bool)
    outside_mask[r0:r1 + 1, c0:c1 + 1] = False

    # Ignore the frame color and the hole color when segmenting pieces.
    piece_mask = outside_mask & (grid != frame_color) & (grid != hole_color)

    pieces = []
    for comp in _connected_components(piece_mask):
        pat, pr0, pc0, ph, pw = _crop_pattern(grid, comp)
        # If the component touches the outside border in a way that makes it
        # just noise/background, keep it anyway; packing will filter unusable.
        pieces.append((pat, ph, pw))

    # If no pieces are found, just return the frame with holes as frame color.
    if not pieces:
        out[hole_mask] = frame_color
        return out

    # Precompute candidate placements for each piece.
    # A placement is (r, c, h, w) inside the hole mask.
    # First try exact sizes; if that fails, allow resizing to nearby sizes.
    rects = _rectangles_in_mask(hole_mask, max_rectangles=5000)

    def valid_place(r, c, h, w):
        if r < 0 or c < 0 or r + h > H or c + w > W:
            return False
        sub = hole_mask[r:r + h, c:c + w]
        return bool(np.all(sub))

    # Candidate placements for each piece, ordered by preference.
    piece_placements = []
    for idx, (pat, ph, pw) in enumerate(pieces):
        candidates = []

        # Exact-size placements.
        for r, c, rh, rw in rects:
            h = rh - r + 1
            w = rw - c + 1
            if h == ph and w == pw and valid_place(r, c, h, w):
                candidates.append((0, r, c, h, w))  # cost 0

        # If no exact candidate, allow resizing to candidate rectangles.
        if not candidates:
            for r, c, rh, rw in rects:
                h = rh - r + 1
                w = rw - c + 1
                if valid_place(r, c, h, w):
                    # Prefer similar aspect and area.
                    area_cost = abs(h * w - ph * pw) / max(1, ph * pw)
                    ar_cost = abs((h / w) - (ph / pw)) / max(1.0, (h / w), (ph / pw))
                    cost = area_cost + 0.5 * ar_cost
                    candidates.append((cost, r, c, h, w))

        # Sort candidates: lower cost first, then top-left order.
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        # Limit candidates to keep search manageable.
        piece_placements.append(candidates[:200])

    # Backtracking search.  Pieces are placed largest-area first.
    order = sorted(range(len(pieces)), key=lambda i: -pieces[i][1] * pieces[i][2])

    occupied = np.zeros((H, W), dtype=bool)
    used_piece = [False] * len(pieces)
    placement = [None] * len(pieces)
    found = False

    # If there are too many pieces, use a greedy fallback after a bounded search.
    def backtrack(pos):
        nonlocal found
        if found:
            return
        if pos == len(order):
            found = True
            return

        pi = order[pos]
        for cost, r, c, h, w in piece_placements[pi]:
            # Check overlap.
            if occupied[r:r + h, c:c + w].any():
                continue
            # Place it.
            occupied[r:r + h, c:c + w] = True
            placement[pi] = (r, c, h, w)
            backtrack(pos + 1)
            occupied[r:r + h, c:c + w] = False
            if found:
                return

    # Bound search by trying only a limited subset if many pieces.
    if len(pieces) <= 12:
        backtrack(0)
    else:
        # Greedy placement for larger inputs.
        for pi in order:
            for cost, r, c, h, w in piece_placements[pi]:
                if not occupied[r:r + h, c:c + w].any():
                    occupied[r:r + h, c:c + w] = True
                    placement[pi] = (r, c, h, w)
                    found = True
                    break
            if not found:
                break

    # If packing failed completely, fall back to filling holes by nearest
    # piece color in a deterministic way.
    if not found or all(p is None for p in placement):
        # Determine dominant non-hole, non-frame colors outside.
        color_counts = Counter()
        outside = outside_mask.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = grid[r, c]
                if val != frame_color and val != hole_color:
                    color_counts[val] += 1

        if color_counts:
            # Fill each hole with the most common outside color.
            fill_color = color_counts.most_common(1)[0][0]
            out[hole_mask] = fill_color
        return out

    # Stamp pieces into the output.
    for pi, p in enumerate(placement):
        if p is None:
            continue
        r, c, h, w = p
        pat = pieces[pi][0]
        if pat.shape != (h, w):
            pat = _resize_pattern(pat, h, w)
        out[r:r + h, c:c + w] = pat

    # Any remaining holes become the frame color.
    out[~occupied] = frame_color
    return out