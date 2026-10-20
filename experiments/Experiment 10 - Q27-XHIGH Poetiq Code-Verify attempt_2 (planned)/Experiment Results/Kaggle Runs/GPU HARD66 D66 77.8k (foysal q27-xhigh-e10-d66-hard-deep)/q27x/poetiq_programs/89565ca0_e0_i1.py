import numpy as np
from collections import defaultdict

def transform(grid: np.ndarray) -> np.ndarray:
    """
    For each colored "container" color, count how many enclosed empty rooms
    belong to it.

    The input contains:
      - 0 as empty space
      - several wall/container colors
      - one scattered "noise/filler" color, which is also used in the output

    The output is a small histogram:
      - one row per container color
      - rows sorted by room count, then by color
      - row i is filled with that color for its room count, and with the
        filler/noise color for the remaining cells
      - width is the maximum room count among all container colors
    """
    if grid.size == 0:
        return np.empty((0, 0), dtype=int)

    h, w = grid.shape
    colors = [c for c in set(grid.ravel().tolist()) if c != 0]
    if not colors:
        return np.zeros((0, 0), dtype=int)

    # ------------------------------------------------------------------
    # 1) Identify the noise/filler color.
    #    It is the color whose largest connected component is smallest.
    # ------------------------------------------------------------------
    def max_component_size(color: int) -> int:
        mask = grid == color
        seen = np.zeros_like(mask, dtype=bool)
        best = 0
        for i in range(h):
            for j in range(w):
                if mask[i, j] and not seen[i, j]:
                    stack = [(i, j)]
                    seen[i, j] = True
                    size = 0
                    while stack:
                        x, y = stack.pop()
                        size += 1
                        for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                            if 0 <= nx < h and 0 <= ny < w:
                                if mask[nx, ny] and not seen[nx, ny]:
                                    seen[nx, ny] = True
                                    stack.append((nx, ny))
                    best = max(best, size)
        return best

    max_comp = {c: max_component_size(c) for c in colors}
    filler_color = min(colors, key=lambda c: (max_comp[c], -np.sum(grid == c), c))

    container_colors = [c for c in colors if c != filler_color]
    if not container_colors:
        return np.zeros((1, 1), dtype=int)

    # ------------------------------------------------------------------
    # 2) Find all enclosed zero regions in the full wall network.
    #    Every non-zero pixel is a wall for the purpose of room separation.
    # ------------------------------------------------------------------
    wall = grid != 0
    visited_zero = np.zeros((h, w), dtype=bool)
    rooms = []

    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0 and not visited_zero[i, j]:
                stack = [(i, j)]
                comp = []
                touches_border = False
                visited_zero[i, j] = True

                while stack:
                    x, y = stack.pop()
                    comp.append((x, y))
                    if x == 0 or y == 0 or x == h - 1 or y == w - 1:
                        touches_border = True

                    for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx, ny] == 0 and not visited_zero[nx, ny]:
                                visited_zero[nx, ny] = True
                                stack.append((nx, ny))

                if not touches_border:
                    rooms.append(comp)

    # ------------------------------------------------------------------
    # 3) For each container color, compute its bounding box and total pixel
    #    count. These are used to attribute each room to its owning color.
    # ------------------------------------------------------------------
    color_info = {}
    for c in container_colors:
        rows = np.where((grid == c).any(axis=1))[0]
        cols = np.where((grid == c).any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        count = int(np.sum(grid == c))
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        color_info[c] = (r0, r1, c0, c1, area, count)

    # ------------------------------------------------------------------
    # 4) Attribute each enclosed room to a container color.
    #
    #    A room is assigned to a color c if:
    #      - the room's bounding box is inside c's bounding box
    #      - the room is adjacent to at least one pixel of c
    #
    #    If multiple colors satisfy this, choose the one with the larger
    #    bounding-box area; ties are broken by total pixel count, then color.
    # ------------------------------------------------------------------
    scores = defaultdict(int)

    for comp in rooms:
        r0 = min(x for x, y in comp)
        r1 = max(x for x, y in comp)
        c0 = min(y for x, y in comp)
        c1 = max(y for x, y in comp)

        boundary = set()
        for x, y in comp:
            for nx, ny in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
                if 0 <= nx < h and 0 <= ny < w:
                    v = grid[nx, ny]
                    if v != 0:
                        boundary.add(v)

        candidates = []
        for c, (cr0, cr1, cc0, cc1, area, count) in color_info.items():
            if c not in boundary:
                continue
            if r0 >= cr0 and r1 <= cr1 and c0 >= cc0 and c1 <= cc1:
                candidates.append((area, count, -c, c))

        if candidates:
            candidates.sort(reverse=True)
            owner = candidates[0][-1]
            scores[owner] += 1
        else:
            # Fallback: use the most frequent non-noise boundary color.
            best = None
            for c in boundary:
                if c == filler_color:
                    continue
                key = (int(np.sum(grid == c)), -c)
                if best is None or key > best[0]:
                    best = (key, c)
            if best is not None:
                scores[best[1]] += 1

    # ------------------------------------------------------------------
    # 5) Build the output histogram.
    # ------------------------------------------------------------------
    items = [(c, scores[c]) for c in scores if scores[c] > 0]
    if not items:
        return np.zeros((1, 1), dtype=int)

    items.sort(key=lambda x: (x[1], x[0]))
    max_count = max(cnt for _, cnt in items)
    out = np.full((len(items), max_count), filler_color, dtype=int)

    for r, (c, cnt) in enumerate(items):
        out[r, :cnt] = c

    return out