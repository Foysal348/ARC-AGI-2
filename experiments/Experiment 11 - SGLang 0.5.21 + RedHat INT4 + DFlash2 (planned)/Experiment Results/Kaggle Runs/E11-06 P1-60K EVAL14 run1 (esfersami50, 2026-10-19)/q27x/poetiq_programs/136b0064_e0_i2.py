import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape

    # Find the vertical divider line of 4s
    divider = None
    for c in range(W):
        if np.all(grid[:, c] == 4):
            divider = c
            break
    if divider is None:
        divider = int(np.argmax([int(np.sum(grid[:, c] == 4)) for c in range(W)]))

    out_w = divider
    out = np.zeros((H, out_w), dtype=int)

    # Place the 5 marker
    ys, xs = np.where(grid == 5)
    r5, c5 = int(ys[0]), int(xs[0])
    p = c5 - divider - 1
    out[r5, p] = 5

    # Extract glyph tiles (connected components) in the left region
    left = grid[:, :out_w]
    visited = np.zeros((H, out_w), dtype=bool)
    tiles = []
    for r in range(H):
        for c in range(out_w):
            if left[r, c] != 0 and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                comp = []
                while q:
                    rr, cc = q.popleft()
                    comp.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if (0 <= nr < H and 0 <= nc < out_w
                                    and not visited[nr, nc] and left[nr, nc] != 0):
                                visited[nr, nc] = True
                                q.append((nr, nc))
                r0 = min(x[0] for x in comp)
                c0 = min(x[1] for x in comp)
                color = int(left[comp[0][0], comp[0][1]])
                sig = frozenset((rr - r0, cc - c0) for rr, cc in comp)
                tiles.append([c0, r0, color, sig])

    # Signature -> (kind, preferred direction, length)
    SIG = {
        frozenset({(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}): ('H', -1, 2),
        frozenset({(0, 0), (0, 2), (1, 1), (2, 1)}): ('V', 0, 2),
        frozenset({(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}): ('H', 1, 3),
        frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 2)}): ('H', -1, 4),
    }

    def params(sig, n):
        if sig in SIG:
            return SIG[sig]
        if n == 4:
            return ('V', 0, 2)
        if n == 5:
            return ('H', 1, 3)
        if n == 6:
            return ('H', -1, 4)
        return ('H', -1, 2)

    # Column-major order: left column top-to-bottom, then right column top-to-bottom
    tiles.sort(key=lambda t: (t[0], t[1]))

    # Draw the continuous path: each segment starts one row below the previous end
    pr, pc = r5, p
    for c0, r0, color, sig in tiles:
        kind, d, L = params(sig, len(sig))
        sr, sc = pr + 1, pc
        if kind == 'V':
            cells = [(sr + k, sc) for k in range(L)]
        else:
            if 0 <= sc + d * (L - 1) < out_w:
                dd = d
            else:
                dd = -d
            cells = [(sr, sc + dd * k) for k in range(L)]
        for rr, cc in cells:
            if 0 <= rr < H and 0 <= cc < out_w:
                out[rr, cc] = color
        pr, pc = cells[-1]

    return out