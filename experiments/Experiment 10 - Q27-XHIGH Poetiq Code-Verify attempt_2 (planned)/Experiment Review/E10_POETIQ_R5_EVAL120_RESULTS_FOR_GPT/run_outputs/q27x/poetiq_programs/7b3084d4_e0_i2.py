import numpy as np
import math


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the 8-connected non-zero pieces into a square.

    Rule inferred from the examples:
      1. Extract all 8-connected non-zero components.
      2. The output size is N x N, where N^2 is the total number of non-zero cells.
      3. The component containing color 5 is placed so that a 5 is at the top-left
         corner of the output.
      4. The other three components are placed at the remaining three corners.
      5. Components may be rotated/reflected. The search chooses a non-overlapping
         exact tiling, preferring fewer/lower-cost transformations.
    """
    input_grid = np.array(grid, dtype=int)

    # Defensive handling for unexpected dimensions.
    if input_grid.ndim == 1:
        input_grid = input_grid.reshape(1, -1)

    if input_grid.size == 0:
        return input_grid.copy()

    H, W = input_grid.shape
    mask = input_grid != 0
    total = int(mask.sum())

    if total == 0:
        return np.zeros_like(input_grid, dtype=int)

    # Output is a square. For valid ARC instances total should be a perfect square.
    N = math.isqrt(total)
    if N * N != total:
        N += 1

    # ------------------------------------------------------------------
    # 8-connected component extraction.
    # ------------------------------------------------------------------
    seen = np.zeros((H, W), dtype=bool)
    comps = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []

            while stack:
                y, x = stack.pop()
                coords.append((y, x))

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < H
                            and 0 <= nx < W
                            and mask[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            stack.append((ny, nx))

            top = min(y for y, x in coords)
            bottom = max(y for y, x in coords)
            left = min(x for y, x in coords)
            right = max(x for y, x in coords)

            piece = np.zeros((bottom - top + 1, right - left + 1), dtype=int)
            for y, x in coords:
                piece[y - top, x - left] = input_grid[y, x]

            comps.append({
                "piece": piece,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "count": len(coords),
                "has5": bool(np.any(piece == 5)),
            })

    K = len(comps)
    if K == 0:
        return np.zeros((N, N), dtype=int)

    # ------------------------------------------------------------------
    # Dihedral transformations: 4 rotations + 4 reflections.
    # Reflections are heavily penalized so rotations are preferred.
    # ------------------------------------------------------------------
    def get_transforms(piece: np.ndarray):
        piece = np.ascontiguousarray(piece, dtype=int)
        results = {}

        def consider(arr, cost):
            arr = np.ascontiguousarray(arr, dtype=int)
            key = (arr.shape, arr.tobytes())
            if key not in results or cost < results[key][1]:
                results[key] = (arr, cost)

        rots = [piece]
        for _ in range(3):
            rots.append(np.rot90(rots[-1]))

        for k, arr in enumerate(rots):
            rot_cost = min(k, 4 - k)  # 0, 1, 2, 1
            consider(arr, rot_cost)
            consider(np.fliplr(arr), 100 + rot_cost)

        vals = list(results.values())
        vals.sort(key=lambda x: (x[1], x[0].shape, x[0].tobytes()))
        return vals

    all_transforms = [get_transforms(comp["piece"]) for comp in comps]

    # ------------------------------------------------------------------
    # Placement helpers.
    # Corner numbering:
    #   0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    # ------------------------------------------------------------------
    def corner_position(arr, corner):
        h, w = arr.shape
        if corner == 0:
            return 0, 0
        if corner == 1:
            return 0, N - w
        if corner == 2:
            return N - h, 0
        return N - h, N - w

    corner_coords = {
        0: (0, 0),
        1: (0, N - 1),
        2: (N - 1, 0),
        3: (N - 1, N - 1),
    }

    def can_place(occupied, arr, r, c):
        h, w = arr.shape
        if r < 0 or c < 0 or r + h > N or c + w > N:
            return False

        piece_mask = arr != 0
        if piece_mask.any() and occupied[r:r + h, c:c + w][piece_mask].any():
            return False

        return True

    def place(out, occupied, arr, r, c):
        h, w = arr.shape
        piece_mask = arr != 0
        if not piece_mask.any():
            return

        out[r:r + h, c:c + w][piece_mask] = arr[piece_mask]
        occupied[r:r + h, c:c + w][piece_mask] = True

    # Precompute which transforms can legally occupy each corner.
    corner_options = [[[] for _ in range(4)] for _ in range(K)]
    for i, transforms in enumerate(all_transforms):
        for arr, cost in transforms:
            h, w = arr.shape
            if h > N or w > N:
                continue

            if arr[0, 0] != 0:
                corner_options[i][0].append((arr, cost))
            if arr[0, w - 1] != 0:
                corner_options[i][1].append((arr, cost))
            if arr[h - 1, 0] != 0:
                corner_options[i][2].append((arr, cost))
            if arr[h - 1, w - 1] != 0:
                corner_options[i][3].append((arr, cost))

        for corner in range(4):
            corner_options[i][corner].sort(
                key=lambda x: (x[1], x[0].shape, x[0].tobytes())
            )

    # ------------------------------------------------------------------
    # Scoring.
    # ------------------------------------------------------------------
    def input_corner(idx):
        comp = comps[idx]
        r_center = (comp["top"] + comp["bottom"]) / 2.0
        c_center = (comp["left"] + comp["right"]) / 2.0

        vertical = "T" if r_center < H / 2.0 else "B"
        horizontal = "L" if c_center < W / 2.0 else "R"
        return vertical + horizontal

    out_corner_names = {0: "TL", 1: "TR", 2: "BL", 3: "BR"}

    def make_score(assigns):
        total_cost = 0
        corner_costs = [0, 0, 0, 0]
        corner_pieces = [0, 0, 0, 0]
        pos_match = 0

        for pi, corner, cost in assigns:
            if 0 <= corner < 4:
                total_cost += cost
                corner_costs[corner] = cost
                corner_pieces[corner] = pi
                if input_corner(pi) == out_corner_names[corner]:
                    pos_match += 1

        # Higher is better.
        return (
            -total_cost,
            tuple(-x for x in corner_costs),
            pos_match,
            tuple(-x for x in corner_pieces),
        )

    best_score = None
    best_out = None

    # The marker is the component containing 5.
    marker_candidates = [i for i, comp in enumerate(comps) if comp["has5"]]
    use_5_marker = bool(marker_candidates)
    if not marker_candidates:
        marker_candidates = list(range(K))

    # The expected task instances have exactly four corner pieces.
    if K == 4 and N > 0:
        for marker_idx in marker_candidates:
            for marker_arr, marker_cost in all_transforms[marker_idx]:
                h, w = marker_arr.shape
                if h > N or w > N:
                    continue

                if use_5_marker:
                    if marker_arr[0, 0] != 5:
                        continue
                else:
                    if marker_arr[0, 0] == 0:
                        continue

                out = np.zeros((N, N), dtype=int)
                occupied = np.zeros((N, N), dtype=bool)
                place(out, occupied, marker_arr, 0, 0)

                remaining = [i for i in range(K) if i != marker_idx]
                assigns = [(marker_idx, 0, marker_cost)]

                def dfs(rem, corners, cur_out, cur_occ, cur_assigns):
                    nonlocal best_score, best_out

                    # Future corners must still be free.
                    for future_corner in corners:
                        fr, fc = corner_coords[future_corner]
                        if cur_occ[fr, fc]:
                            return

                    if not rem:
                        if cur_occ.sum() == N * N:
                            score = make_score(cur_assigns)
                            if best_score is None or score > best_score:
                                best_score = score
                                best_out = cur_out.copy()
                        return

                    corner = corners[0]
                    possible = []

                    for pi in rem:
                        for arr, cost in corner_options[pi][corner]:
                            r, c = corner_position(arr, corner)
                            if can_place(cur_occ, arr, r, c):
                                possible.append((cost, pi, arr, r, c))

                    possible.sort(key=lambda x: (x[0], x[1], x[2].shape, x[2].tobytes()))

                    for cost, pi, arr, r, c in possible:
                        new_out = cur_out.copy()
                        new_occ = cur_occ.copy()
                        place(new_out, new_occ, arr, r, c)

                        new_rem = [x for x in rem if x != pi]
                        cur_assigns.append((pi, corner, cost))

                        dfs(new_rem, corners[1:], new_out, new_occ, cur_assigns)

                        cur_assigns.pop()

                dfs(remaining, [1, 2, 3], out, occupied, assigns)

    if best_out is not None:
        return best_out

    # Defensive fallback: pack all non-zero cells row-major into an N x N square.
    flat = input_grid[mask].flatten()
    out = np.zeros((N, N), dtype=int)
    limit = min(N * N, flat.size)
    out.flat[:limit] = flat[:limit]
    return out