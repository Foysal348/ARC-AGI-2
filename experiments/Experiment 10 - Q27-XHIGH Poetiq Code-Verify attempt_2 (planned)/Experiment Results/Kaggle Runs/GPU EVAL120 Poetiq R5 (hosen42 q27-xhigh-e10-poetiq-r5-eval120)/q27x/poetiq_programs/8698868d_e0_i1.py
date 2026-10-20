import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Combine frame objects with pattern objects.

    Rules:
      - The background is the most frequent color.
      - Non-background connected components are split into frames (larger bboxes)
        and patterns (smaller bboxes).
      - Frames are paired with patterns by sorted color. If a direct same-color
        pairing is possible, the pattern order is cyclically shifted to avoid it.
      - Each output block has the frame's size, a 1-pixel frame-color border,
        and the paired pattern centered/cropped into the interior.
      - Output blocks are laid out in the same spatial arrangement as the frames.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape

    # Background = most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # ---------------------------------------------------------------
    # Connected components of non-background pixels (4-connectivity).
    # ---------------------------------------------------------------
    visited = np.zeros((h, w), dtype=bool)
    comps = []

    for r0 in range(h):
        for c0 in range(w):
            if grid[r0, c0] == bg or visited[r0, c0]:
                continue

            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            visited[r0, c0] = True

            r_min = r_max = r0
            c_min = c_max = c0

            while stack:
                r, c = stack.pop()

                if r < r_min:
                    r_min = r
                if r > r_max:
                    r_max = r
                if c < c_min:
                    c_min = c
                if c > c_max:
                    c_max = c

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "r0": r_min,
                    "c0": c_min,
                    "r1": r_max,
                    "c1": c_max,
                    "h": r_max - r_min + 1,
                    "w": c_max - c_min + 1,
                }
            )

    if not comps:
        return np.zeros_like(grid, dtype=int)

    # Bounding-box area is used to distinguish frames from patterns.
    for comp in comps:
        comp["area"] = comp["h"] * comp["w"]

    comps.sort(key=lambda x: (x["area"], x["h"], x["w"]), reverse=True)

    # The problem structure has equal numbers of frames and patterns.
    half = len(comps) // 2
    if half == 0:
        return np.zeros_like(grid, dtype=int)

    frames = comps[:half]
    patterns = comps[half:]

    # ---------------------------------------------------------------
    # Pair frames and patterns by color order.
    # Cyclically shift the pattern list if needed to avoid same-color pairs.
    # ---------------------------------------------------------------
    frames_sorted = sorted(frames, key=lambda x: (x["color"], x["r0"], x["c0"]))
    patterns_sorted = sorted(patterns, key=lambda x: (x["color"], x["r0"], x["c0"]))

    n = min(len(frames_sorted), len(patterns_sorted))
    shift = 0

    if n > 0:
        for sh in range(n):
            ok = True
            for i in range(n):
                if frames_sorted[i]["color"] == patterns_sorted[(i + sh) % n]["color"]:
                    ok = False
                    break
            if ok:
                shift = sh
                break

    frame_to_pattern = {}
    for i in range(n):
        f = frames_sorted[i]
        p = patterns_sorted[(i + shift) % n]
        frame_to_pattern[id(f)] = p

    # ---------------------------------------------------------------
    # Build one output block for a frame + pattern pair.
    # ---------------------------------------------------------------
    def make_block(frame, pattern):
        fh = frame["h"]
        fw = frame["w"]
        fc = frame["color"]

        block = np.full((fh, fw), fc, dtype=int)

        if pattern is None or fh <= 2 or fw <= 2:
            return block

        ih, iw = fh - 2, fw - 2
        ph, pw = pattern["h"], pattern["w"]
        pc = pattern["color"]

        pat = grid[
            pattern["r0"]:pattern["r1"] + 1,
            pattern["c0"]:pattern["c1"] + 1,
        ]

        # Vertical placement/cropping.
        if ph > ih:
            src_r = (ph - ih) // 2
            sh = ih
            dst_r = 0
        else:
            src_r = 0
            sh = ph
            dst_r = (ih - ph) // 2

        # Horizontal placement/cropping.
        if pw > iw:
            src_c = (pw - iw) // 2
            sw = iw
            dst_c = 0
        else:
            src_c = 0
            sw = pw
            dst_c = (iw - pw) // 2

        pat_sub = pat[src_r:src_r + sh, src_c:src_c + sw]

        interior = np.full((ih, iw), fc, dtype=int)
        interior[dst_r:dst_r + sh, dst_c:dst_c + sw] = np.where(
            pat_sub == pc, pc, fc
        )

        block[1:fh - 1, 1:fw - 1] = interior
        return block

    # ---------------------------------------------------------------
    # Arrange frame blocks in the same row layout as in the input.
    # ---------------------------------------------------------------
    frames_by_row = sorted(frames, key=lambda x: (x["r0"], x["c0"]))
    rows = []

    for f in frames_by_row:
        if not rows:
            rows.append([f])
            continue

        current_row = rows[-1]
        max_bottom = max(x["r1"] + 1 for x in current_row)

        if f["r0"] >= max_bottom:
            rows.append([f])
        else:
            current_row.append(f)

    for row in rows:
        row.sort(key=lambda x: x["c0"])

    strips = []
    for row in rows:
        blocks = []
        for f in row:
            p = frame_to_pattern.get(id(f))
            blocks.append(make_block(f, p))

        if not blocks:
            continue

        # In case block heights differ, pad vertically.
        max_h = max(b.shape[0] for b in blocks)
        padded = []
        for b in blocks:
            if b.shape[0] < max_h:
                pad = np.zeros((max_h - b.shape[0], b.shape[1]), dtype=int)
                b = np.vstack([b, pad])
            padded.append(b)

        strips.append(np.concatenate(padded, axis=1))

    if not strips:
        return np.zeros_like(grid, dtype=int)

    out_h = sum(s.shape[0] for s in strips)
    out_w = max(s.shape[1] for s in strips)

    out = np.zeros((out_h, out_w), dtype=int)
    y = 0
    for s in strips:
        out[y:y + s.shape[0], :s.shape[1]] = s
        y += s.shape[0]

    return out