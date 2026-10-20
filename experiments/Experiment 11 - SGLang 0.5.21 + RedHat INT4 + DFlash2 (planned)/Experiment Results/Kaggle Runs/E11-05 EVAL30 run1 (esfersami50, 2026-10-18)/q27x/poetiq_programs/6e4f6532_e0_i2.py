import numpy as np
from collections import deque, Counter
import itertools


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move each "key" to a matching "lock".

    The rule inferred from the examples:
    - 8 is the body of a key.
    - 9 cells form both the key's internal pattern and the lock pattern.
    - A key is a connected component containing 8 and 9 cells.
    - A lock is a connected component made of 9 cells (and no 8 cells).
    - A key may also contain small "teeth"/tabs of other colors.
    - A key is moved, possibly rotated/reflected, so that its 9 pattern exactly
      matches the 9 pattern of a lock.
    - Among all dihedral placements that match a lock, choose the one whose
      colored teeth best agree with the nearest wall/frame colors in the
      four cardinal directions.
    - Original keys and locks are removed; only background, walls, and the
      placed keys remain.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # -------------------------------------------------------------------------
    # 1. Background color
    # -------------------------------------------------------------------------
    flat = grid.reshape(-1)
    unique, counts = np.unique(flat, return_counts=True)
    bg_color = int(unique[np.argmax(counts)])

    # -------------------------------------------------------------------------
    # 2. Connected components of non-background pixels
    # -------------------------------------------------------------------------
    non_bg = grid != bg_color
    if not non_bg.any():
        return grid.copy()

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def connected_components(mask):
        seen = np.zeros_like(mask, dtype=bool)
        comps = []
        for r in range(H):
            for c in range(W):
                if not mask[r, c] or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                pix = []
                while q:
                    cr, cc = q.popleft()
                    pix.append((cr, cc))
                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(pix)
        return comps

    components = connected_components(non_bg)

    # -------------------------------------------------------------------------
    # 3. Identify static wall/frame components.
    #    Walls are large components that do not contain key/lock colors 8/9.
    # -------------------------------------------------------------------------
    frame_mask = np.zeros((H, W), dtype=bool)
    frame_components = []

    min_frame_size = max(8, int(0.01 * H * W))

    for comp in components:
        colors = {int(grid[r, c]) for r, c in comp}
        has_8 = 8 in colors
        has_9 = 9 in colors

        # Anything containing key/lock material is not treated as a static wall.
        if has_8 or has_9:
            continue

        size = len(comp)
        if size < min_frame_size:
            continue

        rows = [r for r, c in comp]
        cols = [c for r, c in comp]
        h = max(rows) - min(rows) + 1
        w = max(cols) - min(cols) + 1
        fill = size / max(1, h * w)

        # Walls in these tasks are long bars/borders.
        if (
            h >= 0.4 * H
            or w >= 0.4 * W
            or (size >= 0.02 * H * W and fill >= 0.5)
        ):
            frame_components.append(comp)
            for r, c in comp:
                frame_mask[r, c] = True

    frame_set = set(r for r in range(H) for r2, c in ((r, None),) for c in [None]) # placeholder not used
    # (The set is not needed; frame_mask is sufficient.)

    # -------------------------------------------------------------------------
    # 4. Split remaining components into keys and locks.
    #    key: contains 8
    #    lock: contains 9 but no 8
    # -------------------------------------------------------------------------
    keys = []
    locks = []
    movable_components = []

    for comp in components:
        if any(frame_mask[r, c] for r, c in comp):
            # If a component contains both wall pixels and key pixels, this is
            # not expected in the known tasks.  Prefer not to move wall material.
            # Keep it as frame and ignore the rest.
            continue

        colors = {int(grid[r, c]) for r, c in comp}
        has_8 = 8 in colors
        has_9 = 9 in colors

        # Ignore irrelevant background-like components.
        if not has_8 and not has_9:
            movable_components.append(comp)
            continue

        if has_8:
            pixels = [(r, c, int(grid[r, c])) for r, c in comp]
            keys.append(pixels)
            movable_components.append(comp)
        else:
            pixels = [(r, c, int(grid[r, c])) for r, c in comp]
            locks.append(pixels)
            movable_components.append(comp)

    # If there is nothing movable, just return the frame/background grid.
    if not keys or not locks:
        out = grid.copy()
        for comp in movable_components:
            for r, c in comp:
                out[r, c] = bg_color
        return out

    # -------------------------------------------------------------------------
    # 5. Dihedral transformations.
    # -------------------------------------------------------------------------
    transforms = [
        lambda r, c: (r, c),       # identity
        lambda r, c: (c, -r),      # rotate 90
        lambda r, c: (-r, -c),     # rotate 180
        lambda r, c: (-c, r),      # rotate 270
        lambda r, c: (r, -c),      # horizontal flip
        lambda r, c: (-r, c),      # vertical flip
        lambda r, c: (c, r),       # main diagonal
        lambda r, c: (-c, -r),     # anti diagonal
    ]

    def normalize(coords):
        if not coords:
            return None
        rmin = min(r for r, c in coords)
        cmin = min(c for r, c in coords)
        return tuple(sorted((r - rmin, c - cmin) for r, c in coords))

    # -------------------------------------------------------------------------
    # 6. Helper: nearest wall color in a cardinal direction from a bbox.
    # -------------------------------------------------------------------------
    def wall_color_in_direction(side, r0, r1, c0, c1):
        cnt = Counter()

        if side == "top":
            for c in range(c0, c1 + 1):
                r = r0 - 1
                while r >= 0:
                    if frame_mask[r, c]:
                        cnt[int(grid[r, c])] += 1
                        break
                    r -= 1
        elif side == "bottom":
            for c in range(c0, c1 + 1):
                r = r1 + 1
                while r < H:
                    if frame_mask[r, c]:
                        cnt[int(grid[r, c])] += 1
                        break
                    r += 1
        elif side == "left":
            for r in range(r0, r1 + 1):
                c = c0 - 1
                while c >= 0:
                    if frame_mask[r, c]:
                        cnt[int(grid[r, c])] += 1
                        break
                    c -= 1
        elif side == "right":
            for r in range(r0, r1 + 1):
                c = c1 + 1
                while c < W:
                    if frame_mask[r, c]:
                        cnt[int(grid[r, c])] += 1
                        break
                    c += 1

        if not cnt:
            return None
        # Majority color; smaller numeric color breaks ties deterministically.
        return max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def side_tab_colors(pixels):
        r0 = min(r for r, c, _ in pixels)
        r1 = max(r for r, c, _ in pixels)
        c0 = min(c for r, c, _ in pixels)
        c1 = max(c for r, c, _ in pixels)

        tabs = {
            "top": set(),
            "bottom": set(),
            "left": set(),
            "right": set(),
        }

        for r, c, col in pixels:
            if col in (8, 9) or col == bg_color:
                continue
            if r == r0:
                tabs["top"].add(col)
            if r == r1:
                tabs["bottom"].add(col)
            if c == c0:
                tabs["left"].add(col)
            if c == c1:
                tabs["right"].add(col)

        return r0, r1, c0, c1, tabs

    def placement_score(final_pixels):
        if not final_pixels:
            return -10**9

        r0, r1, c0, c1, tabs = side_tab_colors(final_pixels)
        score = 0

        for side in ("top", "bottom", "left", "right"):
            wall = wall_color_in_direction(side, r0, r1, c0, c1)
            tab_set = tabs[side]

            if wall is None:
                # No wall in this direction: no strong constraint.
                continue

            if not tab_set:
                # No tooth on this side: no constraint.
                continue

            if wall in tab_set:
                if len(tab_set) == 1:
                    score += 3
                else:
                    score += 1
            else:
                score -= 3

        return score

    # -------------------------------------------------------------------------
    # 7. Generate candidate placements.
    # -------------------------------------------------------------------------
    candidates = []

    for ki, key in enumerate(keys):
        key_pixels = key
        key_rmin = min(r for r, c, _ in key_pixels)
        key_cmin = min(c for r, c, _ in key_pixels)

        key_9_rel = [
            (r - key_rmin, c - key_cmin)
            for r, c, col in key_pixels
            if col == 9
        ]

        if not key_9_rel:
            continue

        for li, lock in enumerate(locks):
            lock_9 = [(r, c) for r, c, col in lock if col == 9]
            if not lock_9:
                continue

            lock_norm = normalize(lock_9)
            lock_rmin = min(r for r, c in lock_9)
            lock_cmin = min(c for r, c in lock_9)

            for ti, f in enumerate(transforms):
                transformed_9_rel = [f(r, c) for r, c in key_9_rel]
                if normalize(transformed_9_rel) != lock_norm:
                    continue

                s_rmin = min(r for r, c in transformed_9_rel)
                s_cmin = min(c for r, c in transformed_9_rel)

                tr = lock_rmin - s_rmin
                tc = lock_cmin - s_cmin

                final_pixels = []
                valid = True

                for r, c, col in key_pixels:
                    rr, cc = f(r - key_rmin, c - key_cmin)
                    nr = rr + tr
                    nc = cc + tc

                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        valid = False
                        break
                    if frame_mask[nr, nc]:
                        valid = False
                        break

                    final_pixels.append((nr, nc, col))

                if not valid or not final_pixels:
                    continue

                # Sanity check: the 9 pixels must exactly cover the lock 9s.
                final_9 = set((r, c) for r, c, col in final_pixels if col == 9)
                if final_9 != set((r, c) for r, c in lock_9):
                    continue

                score = placement_score(final_pixels)
                pixel_set = frozenset((r, c) for r, c, _ in final_pixels)

                candidates.append(
                    {
                        "key": ki,
                        "lock": li,
                        "transform": ti,
                        "score": score,
                        "pixels": final_pixels,
                        "pixel_set": pixel_set,
                    }
                )

    # -------------------------------------------------------------------------
    # 8. Choose a globally consistent assignment.
    #    For small ARC instances, exhaustive search is safe.
    # -------------------------------------------------------------------------
    if not candidates:
        out = grid.copy()
        for comp in movable_components:
            for r, c in comp:
                out[r, c] = bg_color
        return out

    best_by_key_lock = {}
    for cand in candidates:
        key = cand["key"]
        lock = cand["lock"]
        best_by_key_lock.setdefault((key, lock), []).append(cand)

    for klock in best_by_key_lock:
        best_by_key_lock[klock].sort(key=lambda x: (-x["score"], x["transform"]))
        # Keep only a few best alternatives to keep search tractable.
        best_by_key_lock[klock] = best_by_key_lock[klock][:3]

    n_keys = len(keys)
    n_locks = len(locks)

    best_assignment = []
    best_total = -10**18

    def search(k, used_locks, placed_set, total):
        nonlocal best_assignment, best_total

        if k == n_keys:
            if total > best_total:
                best_total = total
                best_assignment = []
            return

        # If there are more keys than locks, allow a key to be skipped.
        search(k + 1, used_locks, placed_set, total)

        for li in range(n_locks):
            if used_locks & (1 << li):
                continue
            for cand in best_by_key_lock.get((k, li), []):
                if placed_set & cand["pixel_set"]:
                    continue
                search(
                    k + 1,
                    used_locks | (1 << li),
                    placed_set | cand["pixel_set"],
                    total + cand["score"],
                )

    search(0, 0, frozenset(), 0)

    # If the exhaustive search did not find a full assignment because some key
    # has no candidate, fall back to a greedy placement for the possible keys.
    if not best_assignment and candidates:
        chosen = []
        used_locks = 0
        placed = frozenset()

        # Sort all candidates by score descending, then by key/lock for stability.
        sorted_cands = sorted(
            candidates,
            key=lambda x: (-x["score"], x["key"], x["lock"], x["transform"])
        )

        for cand in sorted_cands:
            if used_locks & (1 << cand["lock"]):
                continue
            if placed & cand["pixel_set"]:
                continue
            chosen.append(cand)
            used_locks |= 1 << cand["lock"]
            placed = placed | cand["pixel_set"]

        best_assignment = chosen

    # -------------------------------------------------------------------------
    # 9. Build output.
    # -------------------------------------------------------------------------
    out = grid.copy()

    # Remove all movable objects (keys, locks, and any other non-wall objects).
    for comp in movable_components:
        for r, c in comp:
            if not frame_mask[r, c]:
                out[r, c] = bg_color

    # Place chosen transformed keys.
    for cand in best_assignment:
        for r, c, col in cand["pixels"]:
            if 0 <= r < H and 0 <= c < W and not frame_mask[r, c]:
                out[r, c] = col

    return out