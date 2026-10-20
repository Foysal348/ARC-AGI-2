import numpy as np
import math
from collections import Counter, defaultdict


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a full grid of solid square blocks from:
      - a small palette matrix made of isolated non-background pixels,
      - a visible row of square blocks that shows the block size and pitch.

    The output places one square block at every palette matrix cell.
    A background-colored palette entry means no block is drawn there.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return grid

    H, W = grid.shape

    # Background is the most common color.
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    def divisors(n: int):
        """Return all positive divisors of n."""
        res = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                res.append(i)
                if i != n // i:
                    res.append(n // i)
            i += 1
        return res

    def infer_axis(values):
        """
        Infer a regular 1D lattice from observed coordinates.
        If values are e.g. [1, 3, 5, 7], return [1, 3, 5, 7].
        If one coordinate is missing inside the range, it is still included.
        """
        vals = sorted(set(values))
        if len(vals) <= 1:
            return vals

        step = 0
        for i in range(len(vals) - 1):
            step = math.gcd(step, vals[i + 1] - vals[i])

        if step <= 0:
            return vals

        return list(range(vals[0], vals[-1] + 1, step))

    # ------------------------------------------------------------
    # 1. Connected components
    # ------------------------------------------------------------
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    singleton_pixels = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or grid[r, c] == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pixels = []

            while stack:
                y, x = stack.pop()
                pixels.append((y, x))

                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < H
                        and 0 <= nx < W
                        and not seen[ny, nx]
                        and grid[ny, nx] == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))

            ys = [p[0] for p in pixels]
            xs = [p[1] for p in pixels]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            size = len(pixels)

            comps.append((color, y0, x0, y1, x1, size))

            if size == 1:
                singleton_pixels.append((r, c, color))

    if not singleton_pixels:
        return grid.copy()

    # ------------------------------------------------------------
    # 2. Build the palette matrix from isolated pixels
    # ------------------------------------------------------------
    rows = infer_axis([r for r, _, _ in singleton_pixels])
    cols = infer_axis([c for _, c, _ in singleton_pixels])

    if not rows or not cols:
        return grid.copy()

    R, C = len(rows), len(cols)
    row_idx = {v: i for i, v in enumerate(rows)}
    col_idx = {v: i for i, v in enumerate(cols)}

    palette = np.full((R, C), bg, dtype=grid.dtype)

    for r, c, color in singleton_pixels:
        if r in row_idx and c in col_idx:
            palette[row_idx[r], col_idx[c]] = color

    # ------------------------------------------------------------
    # 3. Extract solid square blocks
    # ------------------------------------------------------------
    blocks = []

    for color, y0, x0, y1, x1, size in comps:
        if size == 1:
            continue

        h = y1 - y0 + 1
        w = x1 - x0 + 1

        # We only use solid rectangular components.
        if size != h * w:
            continue

        if h == w:
            blocks.append((y0, x0, h, color))
        elif w > h and w % h == 0:
            # A horizontal run of touching same-color square blocks.
            for i in range(w // h):
                blocks.append((y0, x0 + i * h, h, color))
        elif h > w and h % w == 0:
            # A vertical run of touching same-color square blocks.
            for i in range(h // w):
                blocks.append((y0 + i * w, x0, w, color))

    if not blocks:
        return grid.copy()

    # The template block size is the most common square side length.
    B = Counter(b[2] for b in blocks).most_common(1)[0][0]
    valid_blocks = [b for b in blocks if b[2] == B]
    if not valid_blocks:
        valid_blocks = blocks

    # The input normally shows one horizontal row of blocks.
    groups = defaultdict(list)
    for y, x, side, color in valid_blocks:
        groups[y].append((x, color))

    # Choose the most informative visible block row.
    y_obs, group = max(
        groups.items(),
        key=lambda kv: (
            len(kv[1]),
            (max(x for x, _ in kv[1]) - min(x for x, _ in kv[1])) if kv[1] else 0,
            -kv[0],
        ),
    )

    obs = sorted(group, key=lambda t: t[0])
    obs_x = [x for x, _ in obs]
    obs_color = [c for _, c in obs]
    m = len(obs)

    # ------------------------------------------------------------
    # 4. Infer pitch and the palette row/column represented by input
    # ------------------------------------------------------------
    max_P = max(H, W, B)

    if C > 1:
        max_P = min(max_P, (W - B) // (C - 1))
    if R > 1:
        max_P = min(max_P, (H - B) // (R - 1))

    if max_P < B:
        max_P = B

    candidates = set()

    if m >= 2:
        D = obs_x[-1] - obs_x[0]
        if D > 0:
            for d in divisors(D):
                if B <= d <= max_P:
                    candidates.add(d)

            for i in range(m - 1):
                d = obs_x[i + 1] - obs_x[i]
                if B <= d <= max_P:
                    candidates.add(d)

    if m == 1 or not candidates:
        for p in range(B, max_P + 1):
            candidates.add(p)

    best = None
    best_score = -1

    for P in sorted(candidates):
        base = obs_x[0]

        # All visible block x-coordinates must align on a grid with step P.
        deltas = []
        ok = True
        for x in obs_x:
            if (x - base) % P != 0:
                ok = False
                break
            deltas.append((x - base) // P)

        if not ok or len(set(deltas)) != m:
            continue

        max_delta = max(deltas)
        if max_delta >= C:
            continue

        for s in range(C - max_delta):
            used_cols = [s + d for d in deltas]
            x_start = base - s * P

            if x_start < 0:
                continue
            if x_start + (C - 1) * P + B > W:
                continue

            for k in range(R):
                # The visible blocks must match the palette row colors.
                if not all(
                    palette[k, col] == color
                    for col, color in zip(used_cols, obs_color)
                ):
                    continue

                y_start = y_obs - k * P

                if y_start < 0:
                    continue
                if y_start + (R - 1) * P + B > H:
                    continue

                # Prefer solutions that do not cover the palette pixels.
                bbox_y1 = y_start + (R - 1) * P + B
                bbox_x1 = x_start + (C - 1) * P + B
                overlap = 0
                for r, c, _ in singleton_pixels:
                    if y_start <= r < bbox_y1 and x_start <= c < bbox_x1:
                        overlap += 1

                score = 0
                score += 1000 * (max_P - P)       # smaller pitch is preferred
                score -= 1000 * overlap           # avoid covering palette dots
                score += min(100, len(obs))       # all exact matches have same obs count

                if score > best_score:
                    best_score = score
                    best = (y_start, x_start, P, R, C, B, palette)

    # ------------------------------------------------------------
    # 5. Fallback if no perfect match is found
    # ------------------------------------------------------------
    if best is None:
        P = B + 1
        if m >= 2:
            diffs = [obs_x[i + 1] - obs_x[i] for i in range(m - 1)]
            if all(d >= B for d in diffs) and len(set(diffs)) == 1:
                P = diffs[0]

        x_start = obs_x[0]

        best_k = 0
        best_match = -1
        for k in range(R):
            match = sum(
                1
                for j in range(min(m, C))
                if palette[k, j] == obs_color[j]
            )
            if match > best_match:
                best_match = match
                best_k = k

        y_start = y_obs - best_k * P

        if y_start < 0:
            y_start = 0
        if y_start + (R - 1) * P + B > H:
            y_start = max(0, H - B - (R - 1) * P)

        if x_start < 0:
            x_start = 0
        if x_start + (C - 1) * P + B > W:
            x_start = max(0, W - B - (C - 1) * P)

        best = (y_start, x_start, P, R, C, B, palette)

    y_start, x_start, P, R, C, B, palette = best

    # ------------------------------------------------------------
    # 6. Draw the complete block grid
    # ------------------------------------------------------------
    out = grid.copy()

    for i in range(R):
        y = y_start + i * P
        y0 = max(0, y)
        y1 = min(H, y + B)

        for j in range(C):
            x = x_start + j * P
            x0 = max(0, x)
            x1 = min(W, x + B)

            if y0 < y1 and x0 < x1:
                out[y0:y1, x0:x1] = int(palette[i, j])

    # Redraw palette dots in case any block area overlapped them.
    for r, c, color in singleton_pixels:
        out[r, c] = color

    return out