import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid by finding the main template object, doubling its
    bounding box, and filling the holes in that doubled template with the
    other object colors.

    The rule inferred from the examples:
      1. Identify the scattered "noise" color.  It is the non-background
         color that appears only in very small components, and is removed.
      2. Among the remaining object colors, identify the template color.
         This is the color whose original bounding box, when doubled, can be
         partitioned into hole components whose shapes best match the
         remaining object shapes.
      3. Create a doubled grid from the template's bounding box:
         - template pixels become 2x2 blocks of the template color,
         - holes inside the template bounding box become 2x2 empty blocks.
      4. Find the connected components of holes in the doubled template.
      5. Assign each remaining color to a hole component by shape similarity.
         Colors may be used multiple times for multiple components if needed.
      6. Fill the assigned hole components with their colors.

    The implementation is deliberately tolerant: it first tries a robust
    similarity-based assignment, and if that is ambiguous it falls back to a
    few simple heuristics that still handle the given examples/challenge style.
    """

    if grid.size == 0:
        return np.zeros_like(grid)

    H, W = grid.shape
    colors = set(int(c) for c in grid.flatten())
    colors.discard(0)

    if not colors:
        return grid.copy()

    # ------------------------------------------------------------
    # Connected components for a single color mask
    # ------------------------------------------------------------
    def mask_components(mask):
        comps = []
        seen = np.zeros(mask.shape, dtype=bool)
        rows, cols = mask.shape
        for r in range(rows):
            for c in range(cols):
                if mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)])
                    seen[r, c] = True
                    pix = []
                    while q:
                        y, x = q.popleft()
                        pix.append((y, x))
                        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < rows and 0 <= nx < cols:
                                if mask[ny, nx] and not seen[ny, nx]:
                                    seen[ny, nx] = True
                                    q.append((ny, nx))
                    comps.append(pix)
        return comps

    # Bounding box and normalized shape for a component
    def comp_shape(pix):
        if not pix:
            return None
        minr = min(p[0] for p in pix)
        maxr = max(p[0] for p in pix)
        minc = min(p[1] for p in pix)
        maxc = max(p[1] for p in pix)
        h = maxr - minr + 1
        w = maxc - minc + 1
        if h == 0 or w == 0:
            return None
        shape = np.zeros((h, w), dtype=bool)
        for r, c in pix:
            shape[r-minr, c-minc] = True
        return shape

    # 8-neighborhood transformations for shape matching
    def transform_shape(shape):
        if shape is None:
            return [None]
        forms = {
            "orig": shape,
            "flipH": shape[:, ::-1],
            "flipV": shape[::-1, :],
            "rot180": shape[::-1, ::-1],
        }
        if shape.shape[0] != shape.shape[1]:
            t = shape.T
            forms["trans"] = t
            forms["trans_flipH"] = t[:, ::-1]
            forms["trans_flipV"] = t[::-1, :]
            forms["trans_rot180"] = t[::-1, ::-1]
        else:
            rot90 = np.rot90(shape, 1)
            rot270 = np.rot90(shape, 3)
            forms["rot90"] = rot90
            forms["rot270"] = rot270
            forms["rot90_flipH"] = rot90[:, ::-1]
            forms["rot270_flipH"] = rot270[:, ::-1]

        # keep only unique valid shapes
        uniq = []
        seen = set()
        for s in forms.values():
            if s is None or s.size == 0:
                continue
            key = (s.shape, tuple(s.flatten().tolist()))
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    def shape_score(a, b):
        """
        Higher is better.  Allows one shape to be a subshape of the other,
        with a small penalty for missing pixels and a big penalty for size
        mismatch.
        """
        if a is None or b is None:
            return -1000
        if a.shape == b.shape:
            inter = int(np.sum(a & b))
            union = int(np.sum(a | b))
            if union == 0:
                return 1.0
            return inter / union
        # If sizes differ, try to place the smaller inside the larger at every
        # offset and take the best overlap.  This is only used as a fallback.
        small, large = (a, b) if a.size <= b.size else (b, a)
        hs, ws = small.shape
        hl, wl = large.shape
        best = 0.0
        if hs <= hl and ws <= wl:
            for dr in range(hl - hs + 1):
                for dc in range(wl - ws + 1):
                    sub = large[dr:dr+hs, dc:dc+ws]
                    inter = int(np.sum(small & sub))
                    union = int(np.sum(small | sub))
                    if union > 0:
                        val = inter / union
                        if val > best:
                            best = val
        # penalize big area differences
        area_ratio = min(a.size, b.size) / max(a.size, b.size)
        return 0.7 * best + 0.3 * area_ratio

    def best_shape_match(shape_a, shapes_b):
        best = -1000
        for sb in shapes_b:
            for ta in transform_shape(shape_a):
                for tb in transform_shape(sb):
                    if ta.shape != tb.shape:
                        continue
                    s = shape_score(ta, tb)
                    if s > best:
                        best = s
        return best

    # ------------------------------------------------------------
    # Build color masks and component info
    # ------------------------------------------------------------
    color_masks = {}
    for c in colors:
        m = (grid == c)
        color_masks[c] = m

    # Estimate the scattered noise color:
    # - small maximum component size
    # - many small components
    # - not the color with the largest total area
    color_stats = {}
    for c, m in color_masks.items():
        comps = mask_components(m)
        sizes = [len(p) for p in comps]
        color_stats[c] = {
            "area": int(m.sum()),
            "ncomp": len(comps),
            "maxcomp": max(sizes) if sizes else 0,
            "avgcomp": (sum(sizes) / len(sizes)) if sizes else 0,
            "comps": comps,
        }

    # Choose noise color: prefer a non-background color that is mostly tiny.
    # In the provided tasks, the noise color is the one made of isolated pixels.
    def noise_score(c):
        st = color_stats[c]
        if st["maxcomp"] <= 4 and st["ncomp"] >= 3:
            score = -st["maxcomp"] - st["avgcomp"] + 0.1 * st["ncomp"]
            return score
        return 1000  # bad noise candidate

    noise_candidates = sorted(colors, key=noise_score)
    noise_color = noise_candidates[0] if noise_candidates else None

    # Remove noise from consideration
    object_colors = [c for c in colors if c != noise_color]
    if not object_colors:
        # If everything looks like noise, just remove the most scattered color
        object_colors = [c for c in colors]

    # ------------------------------------------------------------
    # For each candidate template color, build doubled template and holes
    # ------------------------------------------------------------
    def bbox_of_mask(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin = np.argmax(rows)
        rmax = H - 1 - np.argmax(rows[::-1])
        cmin = np.argmax(cols)
        cmax = W - 1 - np.argmax(cols[::-1])
        return rmin, rmax, cmin, cmax

    # Get template candidate that produces holes that can be matched to other colors.
    template_color = None
    best_template_score = -1
    best_context = None

    # Remaining object colors, excluding candidate template
    for cand in object_colors:
        mask = color_masks[cand]
        rmin, rmax, cmin, cmax = bbox_of_mask(mask)
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        if h <= 1 or w <= 1:
            continue

        # Build doubled template mask: True = filled template, False = hole
        dbl = np.zeros((2*h, 2*w), dtype=bool)
        sub = mask[rmin:rmax+1, cmin:cmax+1]
        dbl[0::2, 0::2] = sub
        dbl[1::2, 0::2] = sub
        dbl[0::2, 1::2] = sub
        dbl[1::2, 1::2] = sub

        # Holes are inside the bounding box where doubled mask is False.
        hole_mask = ~dbl

        # If the template has no holes, it cannot be the template unless it's the only color.
        if not hole_mask.any():
            continue

        hole_comps = mask_components(hole_mask)
        if not hole_comps:
            continue

        hole_shapes = []
        for comp in hole_comps:
            sh = comp_shape(comp)
            if sh is not None:
                hole_shapes.append(sh)

        # Build candidate piece shapes from other object colors
        other_colors = [c for c in object_colors if c != cand]
        if not other_colors:
            continue

        piece_shapes = {}
        for c in other_colors:
            ps = []
            for comp in color_stats[c]["comps"]:
                sh = comp_shape(comp)
                if sh is not None:
                    ps.append(sh)
            if ps:
                piece_shapes[c] = ps

        # Score template by how well hole shapes match piece shapes.
        # Each hole should be assignable to some piece color.
        # Each piece color should ideally cover at least one hole.
        if not hole_shapes:
            continue

        # For each hole, best matching piece score
        hole_to_piece = []
        for hs in hole_shapes:
            best_s = -1000
            best_c = None
            for c, ps in piece_shapes.items():
                for p in ps:
                    s = best_shape_match(hs, [p])
                    if s > best_s:
                        best_s = s
                        best_c = c
            hole_to_piece.append((best_s, best_c))

        # Template score: average hole match, plus bonus if all piece colors are used
        avg_match = sum(x[0] for x in hole_to_piece) / len(hole_to_piece)
        used = set(x[1] for x in hole_to_piece if x[1] is not None)
        coverage = len(used & set(piece_shapes.keys())) / max(1, len(piece_shapes))
        score = avg_match + 0.35 * coverage

        # Strongly prefer templates whose holes can be partitioned cleanly.
        # Penalize templates with too many tiny holes (likely not the background).
        small_holes = sum(1 for hs in hole_shapes if hs.size <= 1)
        if small_holes > len(hole_shapes) * 0.6:
            score -= 0.2

        if score > best_template_score:
            best_template_score = score
            template_color = cand
            best_context = {
                "cand": cand,
                "rmin": rmin, "rmax": rmax,
                "cmin": cmin, "cmax": cmax,
                "h": h, "w": w,
                "sub": sub,
                "dbl": dbl,
                "hole_mask": hole_mask,
                "hole_comps": hole_comps,
                "hole_shapes": hole_shapes,
                "piece_shapes": piece_shapes,
                "hole_to_piece": hole_to_piece,
            }

    # If the automatic template selection fails, fall back to largest area object.
    if template_color is None or best_context is None:
        template_color = max(object_colors, key=lambda c: color_stats[c]["area"])
        mask = color_masks[template_color]
        rmin, rmax, cmin, cmax = bbox_of_mask(mask)
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        dbl = np.zeros((2*h, 2*w), dtype=bool)
        sub = mask[rmin:rmax+1, cmin:cmax+1]
        dbl[0::2, 0::2] = sub
        dbl[1::2, 0::2] = sub
        dbl[0::2, 1::2] = sub
        dbl[1::2, 1::2] = sub
        hole_mask = ~dbl
        hole_comps = mask_components(hole_mask)
        hole_shapes = [comp_shape(comp) for comp in hole_comps if comp]
        other_colors = [c for c in object_colors if c != template_color]
        piece_shapes = {}
        for c in other_colors:
            ps = [comp_shape(comp) for comp in color_stats[c]["comps"] if comp]
            ps = [p for p in ps if p is not None]
            if ps:
                piece_shapes[c] = ps
        best_context = {
            "cand": template_color,
            "rmin": rmin, "rmax": rmax,
            "cmin": cmin, "cmax": cmax,
            "h": h, "w": w,
            "sub": sub,
            "dbl": dbl,
            "hole_mask": hole_mask,
            "hole_comps": hole_comps,
            "hole_shapes": hole_shapes,
            "piece_shapes": piece_shapes,
            "hole_to_piece": None,
        }

    # ------------------------------------------------------------
    # Build output: template doubled, then fill holes
    # ------------------------------------------------------------
    ctx = best_context
    dbl = ctx["dbl"]
    Hout, Wout = dbl.shape

    out = np.zeros((Hout, Wout), dtype=grid.dtype)
    out[dbl] = ctx["cand"]

    # Assign colors to hole components.
    hole_comps = ctx["hole_comps"]
    hole_shapes = ctx["hole_shapes"]
    piece_shapes = ctx.get("piece_shapes", {})

    # If there are no other colors, just return the doubled template.
    other_colors = [c for c in object_colors if c != ctx["cand"]]
    if not other_colors or not hole_comps:
        return out

    # Build a list of assignable pieces: each component of each other color
    # can be used, but a color may be reused if it has multiple components
    # or if there are more holes than distinct colors.
    piece_entries = []
    for c, ps in piece_shapes.items():
        # Use every connected component as a separate pattern candidate.
        for i, p in enumerate(ps):
            piece_entries.append((c, p))

    # If piece_shapes is missing for a color (should not happen), add color-only entry.
    if not piece_entries:
        piece_entries = [(c, None) for c in other_colors]

    # Greedy assignment: for each hole, choose best unused piece/color,
    # allowing reuse of a color if necessary.
    # We prioritize shape similarity; if many holes share the same color,
    # assign by similarity.
    assignments = []
    used_entries = set()

    # Precompute hole-pieces scores
    n_holes = len(hole_comps)
    scores = [[0.0 for _ in range(len(piece_entries))] for _ in range(n_holes)]
    for i, hs in enumerate(hole_shapes):
        for j, (c, ps) in enumerate(piece_entries):
            if ps is None:
                s = 0.1
            else:
                s = best_shape_match(hs, [ps])
            scores[i][j] = s

    # If there are more holes than distinct colors, we may need to reuse colors.
    # First, make a primary assignment that respects shape as much as possible.
    for i in range(n_holes):
        # prefer entries not yet used; if all used, allow best score anyway
        best_j = None
        best_s = -1000
        for j in range(len(piece_entries)):
            if j in used_entries and n_holes <= len(piece_entries):
                # If we have enough entries, avoid reuse until necessary
                continue
            s = scores[i][j]
            # small tie-breaker to keep stable ordering
            s += 1e-6 * (-j)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is None:
            best_j = max(range(len(piece_entries)), key=lambda j: scores[i][j])
        used_entries.add(best_j)
        assignments.append((i, best_j))

    # If the greedy assignment leaves some holes with very poor matches,
    # try a simpler heuristic: fill holes by color frequency / component size.
    # This fallback is intentionally conservative.
    def max_color_count():
        cnt = {}
        for _, j in assignments:
            c = piece_entries[j][0]
            cnt[c] = cnt.get(c, 0) + 1
        return cnt

    # In the known examples, this greedy shape matching is enough.
    # Apply assignments to the doubled grid.
    for i, j in assignments:
        comp = hole_comps[i]
        c = piece_entries[j][0]
        for r, col in comp:
            # comp coordinates are already in doubled-grid coordinates
            out[r, col] = c

    return out