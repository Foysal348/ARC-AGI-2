import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC transformation:

    1. Find the main "window": a complete rectangular frame whose interior
       is filled with a single color. In all provided cases this is the
       color-3 frame, but the code detects it generally.
    2. Output the interior of that window.
    3. For every other rectangular frame visible anywhere in the input,
       infer its full bounding box and draw the portion of its border that
       lies inside the window.
       - Single-color frames are drawn with one color.
       - Two-color frames are drawn with an alternating border.

    The alternating phase is chosen to satisfy as many visible anchor
    pixels as possible over all feasible rectangles and color orientations.
    """

    grid = np.array(grid, dtype=int, copy=True)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # Background is the most frequent border color.
    border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
    bg = int(np.bincount(border, minlength=16).argmax())

    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def components(mask):
        """4-connected components of a boolean mask."""
        seen = np.zeros((H, W), dtype=bool)
        comps = []
        for r in range(H):
            for c in range(W):
                if mask[r, c] and not seen[r, c]:
                    stack = [(r, c)]
                    seen[r, c] = True
                    coords = []
                    while stack:
                        rr, cc = stack.pop()
                        coords.append((rr, cc))
                        for dr, dc in dirs4:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    if coords:
                        comps.append(coords)
        return comps

    def is_complete_frame(bbox, color):
        r0, r1, c0, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            return False
        expected = 2 * h + 2 * w - 4
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        if np.sum(sub == color) != expected:
            return False
        # Every cell of this color in the bbox must lie on the perimeter.
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r, c] == color and not (r == r0 or r == r1 or c == c0 or c == c1):
                    return False
        return True

    def find_window():
        """Find the main window frame."""
        candidates = []

        # Prefer a complete frame whose interior is uniform.
        for color in range(16):
            if color == bg:
                continue
            mask = grid == color
            if not mask.any():
                continue
            for coords in components(mask):
                r0 = min(r for r, _ in coords)
                r1 = max(r for r, _ in coords)
                c0 = min(c for _, c in coords)
                c1 = max(c for _, c in coords)
                if not is_complete_frame((r0, r1, c0, c1), color):
                    continue
                interior = grid[r0 + 1:r1, c0 + 1:c1]
                if interior.size == 0:
                    continue
                uniq = np.unique(interior)
                score = (1 if len(uniq) == 1 else 0, (r1 - r0 + 1) * (c1 - c0 + 1))
                candidates.append((score, (r0, r1, c0, c1)))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Fallback: complete frame of color 3, or largest complete frame.
        fallback = []
        for color in ([3] if np.any(grid == 3) else list(range(16))):
            mask = grid == color
            if not mask.any():
                continue
            for coords in components(mask):
                r0 = min(r for r, _ in coords)
                r1 = max(r for r, _ in coords)
                c0 = min(c for _, c in coords)
                c1 = max(c for _, c in coords)
                if is_complete_frame((r0, r1, c0, c1), color):
                    fallback.append(((r1 - r0 + 1) * (c1 - c0 + 1), (r0, r1, c0, c1)))
        if fallback:
            fallback.sort(key=lambda x: x[0], reverse=True)
            return fallback[0][1]

        return None

    window = find_window()
    if window is None:
        return grid.copy()

    wr0, wr1, wc0, wc1 = window
    ir0, ir1 = wr0 + 1, wr1 - 1
    ic0, ic1 = wc0 + 1, wc1 - 1

    if ir1 < ir0 or ic1 < ic0:
        return np.zeros((0, 0), dtype=int)

    # Base output color: most common color inside the window interior.
    interior = grid[ir0:ir1 + 1, ic0:ic1 + 1]
    vals, counts = np.unique(interior, return_counts=True)
    base = int(vals[np.argmax(counts)])

    out = np.full((ir1 - ir0 + 1, ic1 - ic0 + 1), base, dtype=int)

    # Build mask of candidate object pixels:
    # - not background
    # - not the window frame itself
    # - inside the window, ignore the base fill (but keep non-base pixels)
    window_frame_mask = np.zeros((H, W), dtype=bool)
    window_frame_mask[wr0, wc0:wc1 + 1] = True
    window_frame_mask[wr1, wc0:wc1 + 1] = True
    window_frame_mask[wr0:wr1 + 1, wc0] = True
    window_frame_mask[wr0:wr1 + 1, wc1] = True

    cand = (grid != bg) & (~window_frame_mask)
    cand[ir0:ir1 + 1, ic0:ic1 + 1] = (grid[ir0:ir1 + 1, ic0:ic1 + 1] != base)

    # Group candidate pixels into objects.
    # Each object is a set of connected pieces that can be merged into one
    # rectangle whose perimeter contains all its pixels.
    pieces = []
    for coords in components(cand):
        if len(coords) < 2:
            # Single pixels can be useful for alternating frames, but are
            # often noise; keep only if they are not on the window border.
            if len(coords) == 1:
                r, c = coords[0]
                if grid[r, c] == bg or window_frame_mask[r, c]:
                    continue
            else:
                continue

        arr = np.array(coords, dtype=int)
        r0 = int(arr[:, 0].min())
        r1 = int(arr[:, 0].max())
        c0 = int(arr[:, 1].min())
        c1 = int(arr[:, 1].max())
        pieces.append({
            "coords": arr,
            "bbox": (r0, r1, c0, c1),
            "colors": set(int(grid[r, c]) for r, c in arr),
        })

    def merged_bbox(a, b):
        r0 = min(a["bbox"][0], b["bbox"][0])
        r1 = max(a["bbox"][1], b["bbox"][1])
        c0 = min(a["bbox"][2], b["bbox"][2])
        c1 = max(a["bbox"][3], b["bbox"][3])
        return (r0, r1, c0, c1)

    def merged_coords(a, b):
        return np.vstack((a["coords"], b["coords"]))

    def can_merge(a, b):
        if a is b:
            return False
        colors = a["colors"] | b["colors"]
        if len(colors) > 2:
            return False

        r0, r1, c0, c1 = merged_bbox(a, b)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2:
            return False

        coords = merged_coords(a, b)
        on_perim = (
            (coords[:, 0] == r0) |
            (coords[:, 0] == r1) |
            (coords[:, 1] == c0) |
            (coords[:, 1] == c1)
        )
        if not on_perim.all():
            return False

        # Avoid merging unrelated tiny fragments of different colors that
        # merely share a bounding box.
        if len(colors) == 2:
            # Each color should appear in a plausible alternating border.
            # We do not verify full parity here; the final render will score it.
            pass

        return True

    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            if changed:
                break
            for j in range(i + 1, len(pieces)):
                if can_merge(pieces[i], pieces[j]):
                    pieces[i]["coords"] = merged_coords(pieces[i], pieces[j])
                    pieces[i]["bbox"] = merged_bbox(pieces[i], pieces[j])
                    pieces[i]["colors"] |= pieces[j]["colors"]
                    del pieces[j]
                    changed = True
                    break

    def perimeter_positions(bbox):
        """
        Return a list of (r, c) cells around the rectangle perimeter,
        starting at top-left and moving clockwise.
        """
        r0, r1, c0, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        pts = []

        # top row: (r0, c0..c1)
        for c in range(c0, c1 + 1):
            pts.append((r0, c))
        # right col: (r0+1..r1, c1)
        for r in range(r0 + 1, r1 + 1):
            pts.append((r, c1))
        # bottom row: (r1, c1-1..c0)
        for c in range(c1 - 1, c0 - 1, -1):
            pts.append((r1, c))
        # left col: (r1-1..r0+1, c0)
        for r in range(r1 - 1, r0, -1):
            pts.append((r, c0))

        return pts

    def score_alternating_bbox_color(bbox, c1, c2):
        """
        Score a candidate bbox and color orientation for an alternating frame.
        Higher is better. We prefer:
          - more visible pixels exactly matching the alternating pattern
          - fewer visible mismatches
        """
        r0, r1, c0, c1c = bbox
        h = r1 - r0 + 1
        w = c1c - c0 + 1
        if h < 2 or w < 2:
            return -10**9

        pts = perimeter_positions(bbox)
        # orientation 0: position 0 -> c1, orientation 1: position 0 -> c2
        for start_color in (c1, c2):
            other = c2 if start_color == c1 else c1
            good = 0
            bad = 0
            for idx, (r, c) in enumerate(pts):
                expected = start_color if idx % 2 == 0 else other
                if (r, c) in piece_pixel_set:
                    if grid[r, c] == expected:
                        good += 1
                    elif grid[r, c] == (c1 if expected == c2 else c2):
                        bad += 1
            # Strong penalty for mismatches, reward for matches.
            score = 100 * good - 1000 * bad
            if bad == 0:
                score += 10
            if score > best[0]:
                best[0] = score
                best[1] = start_color
        return best

    # Render objects.
    for piece in pieces:
        r0, r1, c0, c1 = piece["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 1 or w < 1:
            continue

        # Intersection with output window interior.
        or0 = max(r0, ir0)
        or1 = min(r1, ir1)
        oc0 = max(c0, ic0)
        oc1 = min(c1, ic1)
        if or1 < or0 or oc1 < oc0:
            continue

        colors = sorted(piece["colors"])

        if len(colors) == 1:
            color = colors[0]
            for r in range(or0, or1 + 1):
                for c in range(oc0, oc1 + 1):
                    if r == r0 or r == r1 or c == c0 or c == c1:
                        out[r - ir0, c - ic0] = color
        elif len(colors) == 2:
            # For alternating frames, search nearby bbox expansions.
            # Usually the bbox is correct, but a small shift may be needed
            # when one side is completely hidden.
            best_score = -1
            best_bbox = None
            best_start = colors[0]

            # Candidate bboxes: original, and small shifts/expansions.
            candidates = {(r0, r1, c0, c1)}
            for dr0 in (-1, 0, 1):
                for dr1 in (-1, 0, 1):
                    for dc0 in (-1, 0, 1):
                        for dc1 in (-1, 0, 1):
                            nr0, nr1, nc0, nc1 = r0 + dr0, r1 + dr1, c0 + dc0, c1 + dc1
                            if nr0 < 0 or nc0 < 0 or nr1 >= H or nc1 >= W:
                                continue
                            if nr1 <= nr0 or nc1 <= nc0:
                                continue
                            # Candidate must contain all piece pixels.
                            if not (nr0 <= r0 and nr1 >= r1 and nc0 <= c0 and nc1 >= c1):
                                continue
                            candidates.add((nr0, nr1, nc0, nc1))

            piece_pixel_set = set((int(r), int(c)) for r, c in piece["coords"])

            for cand_bbox in candidates:
                rr0, rr1, cc0, cc1 = cand_bbox
                hh = rr1 - rr0 + 1
                ww = cc1 - cc0 + 1
                if hh < 2 or ww < 2:
                    continue

                # All visible pixels must lie on this candidate perimeter.
                ok = True
                for r, c in piece_pixel_set:
                    if not (r == rr0 or r == rr1 or c == cc0 or c == cc1):
                        ok = False
                        break
                if not ok:
                    continue

                # Score both color starts.
                pts = perimeter_positions(cand_bbox)
                for start_color in colors:
                    other = colors[1] if start_color == colors[0] else colors[0]
                    good = 0
                    bad = 0
                    for idx, (r, c) in enumerate(pts):
                        if (r, c) not in piece_pixel_set:
                            continue
                        expected = start_color if idx % 2 == 0 else other
                        if grid[r, c] == expected:
                            good += 1
                        else:
                            bad += 1
                    # Prefer zero bad, then more good, then smaller bbox.
                    score = (bad == 0, good, -(hh * ww))
                    cur = (bad, -good, hh * ww)
                    if best_score == -1:
                        best_score = 0  # placeholder
                    # Use a comparable tuple for best.
                    if best_bbox is None or (bad, -good, hh * ww) < best_score:
                        best_score = (bad, -good, hh * ww)
                        best_bbox = cand_bbox
                        best_start = start_color

            if best_bbox is not None:
                rr0, rr1, cc0, cc1 = best_bbox
                other = colors[1] if best_start == colors[0] else colors[0]
                pts = perimeter_positions(best_bbox)
                # Draw the alternating border inside the window.
                for idx, (r, c) in enumerate(pts):
                    if or0 <= r <= or1 and oc0 <= c <= oc1:
                        color = best_start if idx % 2 == 0 else other
                        out[r - ir0, c - ic0] = color

    # Preserve any non-base pixels that were already inside the window.
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            if grid[r, c] != base:
                out[r - ir0, c - ic0] = int(grid[r, c])

    return out