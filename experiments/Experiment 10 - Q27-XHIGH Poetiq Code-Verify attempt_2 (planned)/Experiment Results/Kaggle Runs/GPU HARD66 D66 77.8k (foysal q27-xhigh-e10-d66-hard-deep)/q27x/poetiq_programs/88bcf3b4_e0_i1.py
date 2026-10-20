import numpy as np
from collections import deque, Counter
from math import gcd

def _neighbors8(p):
    r, c = p
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield (r + dr, c + dc)

def _components(grid):
    H, W = grid.shape
    comp = -np.ones((H, W), dtype=int)
    comps = []
    for i in range(H):
        for j in range(W):
            if comp[i, j] != -1:
                continue
            color = grid[i, j]
            q = deque([(i, j)])
            comp[i, j] = len(comps)
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in _neighbors8((r, c)):
                    if 0 <= nr < H and 0 <= nc < W:
                        if comp[nr, nc] == -1 and grid[nr, nc] == color:
                            comp[nr, nc] = len(comps)
                            q.append((nr, nc))
            comps.append({"color": color, "cells": cells, "id": len(comps) - 1})
    return comps, comp

def _is_straight(cells):
    if len(cells) <= 1:
        return True
    rs = {r for r, c in cells}
    cs = {c for r, c in cells}
    return len(rs) == 1 or len(cs) == 1

def _is_path(cells):
    s = set(cells)
    if len(cells) <= 2:
        return True
    for r, c in cells:
        d = sum(1 for nr, nc in _neighbors8((r, c)) if (nr, nc) in s)
        if d > 2:
            return False
    # A single straight line is not considered a moving snake in these tasks.
    return not _is_straight(cells)

def _endpoints(cells):
    s = set(cells)
    eps = []
    for p in cells:
        d = sum(1 for q in _neighbors8(p) if q in s)
        if d == 1:
            eps.append(p)
    if not eps:
        # Degenerate closed loop: use arbitrary two far cells.
        if cells:
            a = cells[0]
            b = max(cells, key=lambda p: abs(p[0] - a[0]) + abs(p[1] - a[1]))
            return a, b
    if len(eps) == 1:
        # For safety, return the single endpoint with itself.
        return eps[0], eps[0]
    return eps[0], eps[1]

def _orthogonal_adjacent(p, cells):
    r, c = p
    s = set(cells)
    return (r - 1, c) in s or (r + 1, c) in s or (r, c - 1) in s or (r, c + 1) in s

def _anchor_and_target(moving_cells, comps, comp_id):
    e1, e2 = _endpoints(moving_cells)

    # Find endpoint orthogonally adjacent to a straight-line component.
    for A in (e1, e2):
        anchors = []
        for comp in comps:
            if comp["id"] == comp_id:
                continue
            if _is_straight(comp["cells"]) and _orthogonal_adjacent(A, comp["cells"]):
                anchors.append(comp)
        if not anchors:
            continue

        # Choose anchor closest to A.
        anchors.sort(key=lambda comp: min(abs(A[0] - r) + abs(A[1] - c) for r, c in comp["cells"]))
        anchor = anchors[0]

        # Determine away side from the anchor line.
        ar, ac = A
        if len({r for r, c in anchor["cells"]}) == 1:
            # vertical anchor line
            line_col = anchor["cells"][0][1]
            if ac < line_col:
                side = "left"
            elif ac > line_col:
                side = "right"
            else:
                return None
        else:
            # horizontal anchor line
            line_row = anchor["cells"][0][0]
            if ar < line_row:
                side = "up"
            elif ar > line_row:
                side = "down"
            else:
                return None

        # Choose target: nearest non-moving, non-anchor component.
        target = None
        best = None
        for comp in comps:
            if comp["id"] == comp_id or comp["id"] == anchor["id"]:
                continue
            d = min(abs(ar - r) + abs(ac - c) for r, c in comp["cells"])
            if best is None or d < best:
                best = d
                target = comp
        if target is None:
            continue

        return A, e1 if A == e2 else e2, anchor, target, side
    return None

def _line_path(a, b, include_start=True):
    """8-neighbor straight/step path from a to b, preferring diagonal moves."""
    a = tuple(a)
    b = tuple(b)
    path = [a] if include_start else []
    r, c = a
    tr, tc = b
    while (r, c) != (tr, tc):
        dr = 0 if r == tr else (1 if tr > r else -1)
        dc = 0 if c == tc else (1 if tc > c else -1)
        r += dr
        c += dc
        path.append((r, c))
    return path

def _side_chain(target_cells, side):
    """
    Boundary cells on the requested side of target, ordered from the end
    toward which the path should move.
    """
    cells = set(target_cells)
    if side in ("left", "right"):
        rows = sorted({r for r, c in cells})
        chain = []
        for r in rows:
            cols = [c for rr, c in cells if rr == r]
            if side == "left":
                cmax = max(cols)
                # closest non-target cell to the left, 8-adjacent to target
                cand = cmax - 1
                if cand not in cells and any((r + dr, cand + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                    chain.append((r, cand))
            else:
                cmin = min(cols)
                cand = cmin + 1
                if cand not in cells and any((r + dr, cand + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                    chain.append((r, cand))
        # If missing due to gaps, keep only valid adjacent cells.
        valid = []
        for p in chain:
            r, c = p
            if (r, c) in cells:
                continue
            if any((r + dr, c + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                valid.append(p)
        chain = valid
        # Return ordered from bottom to top if going up, else top to bottom.
        return chain  # caller reverses based on direction

    else:
        cols = sorted({c for r, c in cells})
        chain = []
        for c in cols:
            rows = [r for r, cc in cells if cc == c]
            if side == "up":
                rmax = max(rows)
                cand = rmax - 1
                if cand not in cells and any((cand + dr, c + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                    chain.append((cand, c))
            else:
                rmin = min(rows)
                cand = rmin + 1
                if cand not in cells and any((cand + dr, c + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                    chain.append((cand, c))
        valid = []
        for p in chain:
            r, c = p
            if (r, c) in cells:
                continue
            if any((r + dr, c + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)):
                valid.append(p)
        return valid

def _generate_path(A, B, anchor, target, side, length, grid):
    H, W = grid.shape
    A = tuple(A)
    ar, ac = A

    # Determine vertical/horizontal travel direction.
    tc = sum(r for r, c in target["cells"]) / len(target["cells"])
    tcc = sum(c for r, c in target["cells"]) / len(target["cells"])

    if side in ("left", "right"):
        if tc < ar:
            vdir = -1  # target above
        elif tc > ar:
            vdir = 1   # target below
        else:
            vdir = -1 if ar > 0 else 1

        chain = _side_chain(target["cells"], side)
        # For left/right side, chain rows are sorted ascending.
        if vdir == -1:
            # moving up: start at bottom-most chain cell
            if chain:
                start = chain[-1]
                chain = list(reversed(chain))
            else:
                start = None
        else:
            if chain:
                start = chain[0]
            else:
                start = None

        if start is None:
            # Fallback: straight path toward target center.
            path = _line_path(A, (int(tc), int(tcc)))
            return path[:length]

        # Approach to start boundary cell.
        approach = _line_path(A, start)
        # Follow chain.
        path = approach
        for p in chain:
            if p == path[-1]:
                continue
            # Small direct step from current to chain point.
            seg = _line_path(path[-1], p, include_start=False)
            path.extend(seg)
            if len(path) > length:
                break

        # Leave diagonally away from target side.
        if side == "left":
            leave = (vdir, 1)
        else:
            leave = (vdir, -1)

        r, c = path[-1]
        while len(path) < length:
            nr, nc = r + leave[0], c + leave[1]
            if not (0 <= nr < H and 0 <= nc < W):
                break
            path.append((nr, nc))
            r, c = nr, nc
        return path[:length]

    else:
        # side is up/down
        if tcc < ac:
            hdir = -1  # target left
        elif tcc > ac:
            hdir = 1   # target right
        else:
            hdir = -1 if ac > 0 else 1

        chain = _side_chain(target["cells"], side)
        # chain columns are sorted ascending
        if hdir == -1:
            if chain:
                start = chain[-1]
                chain = list(reversed(chain))
            else:
                start = None
        else:
            if chain:
                start = chain[0]
            else:
                start = None

        if start is None:
            path = _line_path(A, (int(tc), int(tcc)))
            return path[:length]

        approach = _line_path(A, start)
        path = approach
        for p in chain:
            if p == path[-1]:
                continue
            seg = _line_path(path[-1], p, include_start=False)
            path.extend(seg)
            if len(path) > length:
                break

        if side == "up":
            leave = (-1, hdir)
        else:
            leave = (1, hdir)

        r, c = path[-1]
        while len(path) < length:
            nr, nc = r + leave[0], c + leave[1]
            if not (0 <= nr < H and 0 <= nc < W):
                break
            path.append((nr, nc))
            r, c = nr, nc
        return path[:length]

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform each moving snake-like component by anchoring it at its
    orthogonally adjacent straight-line endpoint, then wrapping around the
    nearest target component on the side away from the anchor line.
    """
    grid = np.array(grid)
    H, W = grid.shape

    # Determine background color as the most frequent pixel value.
    flat = grid.flatten()
    bg = Counter(flat).most_common(1)[0][0]

    comps, comp_id = _components(grid)

    # Candidate moving components: 8-connected, path-like, non-straight.
    candidates = []
    for comp in comps:
        if comp["color"] == bg:
            continue
        if not _is_path(comp["cells"]):
            continue
        candidates.append(comp)

    if not candidates:
        return grid

    # Sort by size descending so larger snakes take precedence if overlapping.
    candidates.sort(key=lambda x: len(x["cells"]), reverse=True)

    out = grid.copy()
    used_cells = set()

    for comp in candidates:
        info = _anchor_and_target(comp["cells"], comps, comp["id"])
        if info is None:
            continue
        A, B, anchor, target, side = info
        length = len(comp["cells"])
        if length <= 1:
            continue

        path = _generate_path(A, B, anchor, target, side, length, out)
        if not path:
            continue

        # Erase original moving component.
        for r, c in comp["cells"]:
            if out[r, c] == comp["color"]:
                out[r, c] = bg

        # Draw new path, but do not overwrite anchors/targets/other static objects.
        for r, c in path:
            if 0 <= r < H and 0 <= c < W:
                # Keep anchor/target cells intact.
                if (r, c) in set(target["cells"]) or (r, c) in set(anchor["cells"]):
                    continue
                out[r, c] = comp["color"]

    return out