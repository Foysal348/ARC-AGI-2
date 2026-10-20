import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble framed tiles.

    Heuristic rule:
      - Background is the most frequent color.
      - Square monochrome objects are extracted using 4-connectivity per color.
      - Largest square size S with available (S-2) squares gives:
          * frames = S x S objects
          * patterns = (S-2) x (S-2) objects
      - Each frame has a "notch count": number of non-frame pixels inside its bbox.
      - Frames are ordered by notch count, then input row-major order.
      - Patterns are ordered left-to-right, then top-to-bottom.
      - Frame notch count k selects pattern number k in that pattern order.
        If that pattern is already used, take the next unused one.
      - Output is the compact grid of frames, each filled with its pattern.
    """
    if grid is None or grid.size == 0:
        return grid

    grid = np.array(grid, copy=True)
    h, w = grid.shape

    # Background color: most frequent color.
    bg = int(np.bincount(grid.astype(int).ravel()).argmax())

    # ------------------------------------------------------------------
    # Extract square monochrome components.
    # Components are connected by 4-neighbors of the SAME color.
    # ------------------------------------------------------------------
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True

            minr = maxr = r
            minc = maxc = c
            pixel_count = 0

            while stack:
                cr, cc = stack.pop()
                pixel_count += 1

                minr = min(minr, cr)
                maxr = max(maxr, cr)
                minc = min(mincc, cc) if (mincc := cc) is not None else cc
                maxc = max(maxc, cc)

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not seen[nr, nc] and grid[nr, nc] == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

            height = maxr - minr + 1
            width = maxc - minc + 1

            if height != width:
                continue

            # Keep only reasonably solid square objects.
            if pixel_count < 0.35 * height * width:
                continue

            components.append({
                "uid": len(components),
                "color": color,
                "minr": minr,
                "minc": minc,
                "maxr": maxr,
                "maxc": maxc,
                "side": height,
            })

    if not components:
        return grid

    # ------------------------------------------------------------------
    # Choose frame size.
    # Frames are the largest square size S for which (S-2) squares exist.
    # ------------------------------------------------------------------
    sizes = sorted({comp["side"] for comp in components}, reverse=True)
    frame_size = None
    frames = []
    patterns = []

    for s in sizes:
        if s < 3:
            continue
        candidate_frames = [comp for comp in components if comp["side"] == s]
        candidate_patterns = [comp for comp in components if comp["side"] == s - 2]
        if candidate_frames and candidate_patterns:
            frame_size = s
            frames = candidate_frames
            patterns = candidate_patterns
            break

    if frame_size is None or not frames or not patterns:
        return grid

    # Notch count for each frame: non-frame-color pixels in its bbox.
    for f in frames:
        sub = grid[f["minr"]:f["maxr"] + 1, f["minc"]:f["maxc"] + 1]
        f["notches"] = int(np.sum(sub != f["color"]))

    # ------------------------------------------------------------------
    # Assign patterns to frames.
    # ------------------------------------------------------------------
    patterns.sort(key=lambda p: (p["minc"], p["minr"]))
    frames_sorted = sorted(frames, key=lambda f: (f["notches"], f["minr"], f["minc"]))

    assignment = {}
    used_pattern_indices = set()
    n_patterns = len(patterns)

    for f in frames_sorted:
        if n_patterns == 0:
            break

        # 1-based notch count -> 0-based index, clamped.
        idx = f["notches"] - 1
        if idx < 0:
            idx = 0
        if idx >= n_patterns:
            idx = n_patterns - 1

        chosen = None
        for offset in range(n_patterns):
            cand = (idx + offset) % n_patterns
            if cand not in used_pattern_indices:
                chosen = cand
                break

        if chosen is None:
            break

        assignment[f["uid"]] = patterns[chosen]
        used_pattern_indices.add(chosen)

    # Fallback for any unassigned frames: use nearest unused pattern.
    unassigned = [f for f in frames if f["uid"] not in assignment]
    unused_indices = [i for i in range(n_patterns) if i not in used_pattern_indices]

    if unassigned and unused_indices:
        for f in sorted(unassigned, key=lambda x: (x["minr"], x["minc"])):
            fr = (f["minr"] + f["maxr"] + 1) / 2.0
            fc = (f["minc"] + f["maxc"] + 1) / 2.0

            def dist(i):
                p = patterns[i]
                pr = (p["minr"] + p["maxr"] + 1) / 2.0
                pc = (p["minc"] + p["maxc"] + 1) / 2.0
                return (fr - pr) ** 2 + (fc - pc) ** 2

            best = min(unused_indices, key=dist)
            assignment[f["uid"]] = patterns[best]
            used_pattern_indices.add(best)
            unused_indices.remove(best)

    # ------------------------------------------------------------------
    # Build compact output grid.
    # ------------------------------------------------------------------
    def cluster_values(values, threshold):
        values = sorted(set(values))
        if not values:
            return []
        groups = [[values[0]]]
        for v in values[1:]:
            if v - groups[-1][-1] <= threshold:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [g[0] for g in groups]

    threshold = max(1, frame_size // 2)
    rows = cluster_values([f["minr"] for f in frames], threshold)
    cols = cluster_values([f["minc"] for f in frames], threshold)

    if not rows or not cols:
        return grid

    R = len(rows)
    C = len(cols)
    out = np.zeros((R * frame_size, C * frame_size), dtype=grid.dtype)

    def nearest_index(value, arr):
        return int(np.argmin(np.abs(np.asarray(arr) - value)))

    def fit_pattern(arr, target, background, dtype):
        if target <= 0:
            return np.zeros((0, 0), dtype=dtype)

        ph, pw = arr.shape

        if ph > target:
            top = (ph - target) // 2
            arr = arr[top:top + target, :]
            ph = target

        if pw > target:
            left = (pw - target) // 2
            arr = arr[:, left:left + target]
            pw = target

        if ph < target or pw < target:
            tmp = np.full((target, target), background, dtype=dtype)
            top = (target - ph) // 2
            left = (target - pw) // 2
            tmp[top:top + ph, left:left + pw] = arr
            arr = tmp

        return arr

    target = frame_size - 2

    for f in frames:
        ri = nearest_index(f["minr"], rows)
        ci = nearest_index(f["minc"], cols)

        tile = np.full((frame_size, frame_size), f["color"], dtype=grid.dtype)
        pat = assignment.get(f["uid"])

        if pat is not None and target > 0:
            psub = grid[pat["minr"]:pat["maxr"] + 1,
                       pat["minc"]:pat["maxc"] + 1].copy()
            psub = fit_pattern(psub, target, bg, grid.dtype)
            psub = np.where(psub == bg, f["color"], psub)
            tile[1:-1, 1:-1] = psub

        out[ri * frame_size:(ri + 1) * frame_size,
            ci * frame_size:(ci + 1) * frame_size] = tile

    return out