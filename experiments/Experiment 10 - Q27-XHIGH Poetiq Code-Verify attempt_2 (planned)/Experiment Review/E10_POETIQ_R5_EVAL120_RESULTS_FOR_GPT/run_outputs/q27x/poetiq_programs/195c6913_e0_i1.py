import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the thick target-colored stroke into a thin diagonal-pattern stroke.

    Rules inferred from the examples:
    - The border color is the background.
    - The other most frequent color is the target stroke color.
    - The top 2x2 blocks give the repeating pattern sequence.
    - The first left-border non-background/non-target cell anchors the phase.
    - The remaining small color is the endpoint marker color.
    - A 4-connected skeleton of the target region is extracted.
    - Skeleton cells are overwritten with the repeating diagonal pattern.
    - Skeleton leaf endpoints are extended by one cell, and that extension cell
      is marked with the endpoint color.
    """
    if grid.size == 0:
        return grid.copy()

    h, w = grid.shape
    out = grid.copy()

    flat = grid.reshape(-1)
    colors, counts = np.unique(flat, return_counts=True)
    freq = dict(zip(colors.tolist(), counts.tolist()))

    # Background: most common border color.
    border = np.concatenate([
        grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
    ])
    border_colors, border_counts = np.unique(border, return_counts=True)
    bg = int(border_colors[np.argmax(border_counts)])

    # Target: most common non-background color.
    target = None
    best = -1
    for c, n in freq.items():
        if c != bg and n > best:
            best = n
            target = int(c)
    if target is None:
        return out

    # Find top 2x2 marker blocks, preserving order and repeats.
    seq = []
    seen = np.zeros((h, w), dtype=bool)
    for r in range(0, min(4, h - 1)):
        for c in range(0, w - 1):
            val = int(grid[r, c])
            if val in (bg, target):
                continue
            if (grid[r, c] == grid[r, c + 1] ==
                grid[r + 1, c] == grid[r + 1, c + 1] == val):
                if not seen[r:r + 2, c:c + 2].any():
                    seq.append(val)
                    seen[r:r + 2, c:c + 2] = True

    if not seq:
        return out

    # Endpoint color: first color not background, target, or in seq.
    endpoint = None
    for c in colors:
        c = int(c)
        if c not in (bg, target) and c not in seq:
            endpoint = c
            break
    if endpoint is None:
        endpoint = target

    # Phase anchor: first left-border seed matching seq[0] if possible.
    seed_row = 0
    for r in range(h):
        if int(grid[r, 0]) not in (bg, target) and int(grid[r, 0]) == seq[0]:
            seed_row = r
            break
    else:
        for r in range(h):
            if int(grid[r, 0]) not in (bg, target):
                seed_row = r
                break

    N = len(seq)
    offset = seed_row % N

    def pattern_color(r, c):
        return seq[((c - r + offset) % N + N) % N]

    # --- Zhang-Suen thinning (4-connected-friendly variant) ---
    mask = (grid == target).astype(np.uint8)

    if mask.sum() == 0:
        return out

    def neighbors8(x, y):
        idx = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w:
                    idx.append((nx, ny))
        return idx

    def count_8(x, y):
        return sum(1 for nx, ny in neighbors8(x, y) if mask[nx, ny] == 1)

    def transitions(x, y):
        vals = [
            mask[x, y + 1], mask[x - 1, y + 1],
            mask[x - 1, y], mask[x - 1, y - 1],
            mask[x, y - 1], mask[x + 1, y - 1],
            mask[x + 1, y], mask[x + 1, y + 1]
        ]
        t = 0
        for i in range(8):
            if vals[i] == 0 and vals[(i + 1) % 8] == 1:
                t += 1
        return t

    def thin_iter(img):
        h2, w2 = img.shape
        to_del = set()
        for step in (1, 2):
            for x in range(1, h2 - 1):
                for y in range(1, w2 - 1):
                    if img[x, y] == 0:
                        continue
                    if (x, y) in to_del:
                        continue
                    P2 = img[x, y + 1]
                    P3 = img[x - 1, y + 1]
                    P4 = img[x - 1, y]
                    P5 = img[x - 1, y - 1]
                    P6 = img[x, y - 1]
                    P7 = img[x + 1, y - 1]
                    P8 = img[x + 1, y]
                    P9 = img[x + 1, y + 1]

                    B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
                    A = 0
                    if P2 == 0 and P3 == 1: A += 1
                    if P3 == 0 and P4 == 1: A += 1
                    if P4 == 0 and P5 == 1: A += 1
                    if P5 == 0 and P6 == 1: A += 1
                    if P6 == 0 and P7 == 1: A += 1
                    if P7 == 0 and P8 == 1: A += 1
                    if P8 == 0 and P9 == 1: A += 1
                    if P9 == 0 and P2 == 1: A += 1

                    if step == 1:
                        if (2 <= B <= 6 and A == 1 and
                                P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
                            to_del.add((x, y))
                    else:
                        if (2 <= B <= 6 and A == 1 and
                                P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                            to_del.add((x, y))
        for x, y in to_del:
            img[x, y] = 0

    skeleton = mask.copy()
    while True:
        before = skeleton
        thin_iter(skeleton)
        if (skeleton == before).all():
            break

    # If thinning leaves nothing (possible in tiny shapes), fall back to target.
    if skeleton.sum() == 0:
        skeleton = mask.copy()

    # Build 4-connected skeleton graph and find leaves.
    sk_coords = [(r, c) for r in range(h) for c in range(w) if skeleton[r, c] == 1]
    coord_set = set(sk_coords)

    deg = {}
    for r, c in sk_coords:
        d = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in coord_set:
                d += 1
        deg[(r, c)] = d

    # Add one-step extensions at leaves and mark endpoint color.
    for (r, c), d in deg.items():
        if d != 1:
            continue
        # The only neighbor direction defines the inward direction.
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in coord_set:
                # Outward direction is opposite.
                nr, nc = r - dr, c - dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = endpoint
                break

    # Paint skeleton cells with diagonal pattern.
    for r, c in sk_coords:
        out[r, c] = pattern_color(r, c)

    return out