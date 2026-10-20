import numpy as np
from itertools import product

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid into the assembled square.

    The operation is:
    1. Extract all 8-connected non-zero components.
    2. Let N be the integer square root of the total number of non-zero cells.
    3. The component containing color 5 is placed at the top-left corner, with
       its 5 cell at output (0, 0), choosing a valid dihedral transform.
    4. Two other components are placed at the bottom-left and bottom-right
       corners.  A component may be rotated/reflected, but its assigned corner
       cell must be occupied.
    5. The remaining empty cells are filled with the color of the leftover
       component.  If the leftover component's shape exactly matches the empty
       shape under some transform, its colors are used directly; otherwise the
       dominant color of that component fills the empty area.

    This reproduces the given examples, where the pieces interlock to form a
    zero-free square.
    """

    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape
    non_zero = grid > 0
    total = int(non_zero.sum())

    # If there are no pieces, return zeros of the same shape.
    if total == 0:
        return np.zeros_like(grid, dtype=int)

    # The outputs in this task are squares whose area is the total cell count.
    N = int(round(np.sqrt(total)))
    if N * N != total:
        # Fallback: minimal square that can contain all cells.
        N = int(np.ceil(np.sqrt(total)))

    # ------------------------------------------------------------------
    # 8-connected component labeling.
    # ------------------------------------------------------------------
    seen = np.zeros_like(grid, dtype=bool)
    components = []

    for r in range(H):
        for c in range(W):
            if not non_zero[r, c] or seen[r, c]:
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
                        if 0 <= ny < H and 0 <= nx < W:
                            if non_zero[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))

            top = min(y for y, x in coords)
            left = min(x for y, x in coords)
            bottom = max(y for y, x in coords)
            right = max(x for y, x in coords)
            ph, pw = bottom - top + 1, right - left + 1
            piece = np.zeros((ph, pw), dtype=int)
            for y, x in coords:
                piece[y - top, x - left] = grid[y, x]

            components.append({
                "piece": piece,
                "top": top,
                "left": left,
                "count": len(coords),
                "has_5": int((piece == 5).sum()) > 0
            })

    # Deterministic component order: top, then left.
    components.sort(key=lambda cp: (cp["top"], cp["left"]))

    # ------------------------------------------------------------------
    # Dihedral transforms of a piece.
    # ------------------------------------------------------------------
    def all_transforms(piece: np.ndarray):
        base = piece
        seen_shapes = set()
        out = []
        for _ in range(4):
            for arr in (base, np.fliplr(base)):
                key = tuple(map(tuple, arr))
                if key not in seen_shapes:
                    seen_shapes.add(key)
                    out.append(np.array(arr, dtype=int))
            base = np.rot90(base)
        return out

    def place_no_overlap(occupied: np.ndarray, piece: np.ndarray, r0: int, c0: int):
        h, w = piece.shape
        if r0 < 0 or c0 < 0 or r0 + h > N or c0 + w > N:
            return False

        mask = piece > 0
        if occupied[r0:r0 + h, c0:c0 + w][mask].any():
            return False
        return True

    def put(occupied: np.ndarray, piece: np.ndarray, r0: int, c0: int):
        h, w = piece.shape
        occupied[r0:r0 + h, c0:c0 + w] |= (piece > 0)

    def connected8(mask: np.ndarray):
        if not mask.any():
            return False
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        start = np.argwhere(mask)[0]
        stack = [tuple(start)]
        seen[start[0], start[1]] = True
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        return seen.sum() == mask.sum()

    def crop_mask(mask: np.ndarray):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        return mask[r0:r1 + 1, c0:c1 + 1]

    def shape_matches_any_transform(mask: np.ndarray, piece: np.ndarray):
        target = crop_mask(mask)
        th, tw = target.shape
        for t in all_transforms((piece > 0).astype(int)):
            if t.shape != target.shape:
                continue
            if np.array_equal(t > 0, target):
                return True
        return False

    def dominant_color(piece: np.ndarray) -> int:
        vals, counts = np.unique(piece[piece > 0], return_counts=True)
        return int(vals[np.argmax(counts)])

    # ------------------------------------------------------------------
    # Choose the top-left component.  Prefer the one containing 5.
    # ------------------------------------------------------------------
    marker_index = None
    for i, cp in enumerate(components):
        if cp["has_5"]:
            marker_index = i
            break

    if marker_index is None:
        # Fallback: try every component as the top-left anchor.
        top_candidates = list(range(len(components)))
    else:
        top_candidates = [marker_index]

    best_score = None
    best_output = None

    for top_idx in top_candidates:
        top_comp = components[top_idx]
        top_piece = top_comp["piece"]

        # Try all transforms of the top-left piece.
        for top_t in all_transforms(top_piece):
            th, tw = top_t.shape
            if th > N or tw > N:
                continue

            # The anchor must occupy the top-left corner.  If there is a 5,
            # it should be exactly at (0, 0).
            if top_comp["has_5"]:
                if top_t[0, 0] != 5:
                    continue
            else:
                if top_t[0, 0] == 0:
                    continue

            occupied = np.zeros((N, N), dtype=bool)
            put(occupied, top_t, 0, 0)

            remaining = [i for i in range(len(components)) if i != top_idx]
            if len(remaining) < 3:
                continue

            # Try choosing two of the remaining pieces for bottom-left and
            # bottom-right.  The third is the filler.
            for left_idx in remaining:
                left_comp = components[left_idx]
                for right_idx in [x for x in remaining if x != left_idx]:
                    right_comp = components[right_idx]
                    filler_idx = [x for x in remaining if x not in (left_idx, right_idx)][0]
                    filler_comp = components[filler_idx]

                    for left_t in all_transforms(left_comp["piece"]):
                        lh, lw = left_t.shape
                        if lh > N or lw > N:
                            continue
                        # Bottom-left corner must be occupied.
                        if left_t[lh - 1, 0] == 0:
                            continue
                        lr, lc = N - lh, 0
                        if not place_no_overlap(occupied, left_t, lr, lc):
                            continue

                        temp_occ = occupied.copy()
                        put(temp_occ, left_t, lr, lc)

                        for right_t in all_transforms(right_comp["piece"]):
                            rh, rw = right_t.shape
                            if rh > N or rw > N:
                                continue
                            # Bottom-right corner must be occupied.
                            if right_t[rh - 1, rw - 1] == 0:
                                continue
                            rr, rc = N - rh, N - rw
                            if not place_no_overlap(temp_occ, right_t, rr, rc):
                                continue

                            final_occ = temp_occ.copy()
                            put(final_occ, right_t, rr, rc)

                            empty = ~final_occ
                            if empty.sum() != filler_comp["count"]:
                                continue

                            # The filler should be the top-right region.
                            if final_occ[0, N - 1]:
                                continue

                            # Require both placed pieces and filler to be connected.
                            if not connected8(final_occ):
                                continue
                            if not connected8(empty):
                                continue

                            # Score this assembly.
                            filler_match = 1 if shape_matches_any_transform(empty, filler_comp["piece"]) else 0

                            # Prefer wide bottom-left and tall bottom-right shapes.
                            left_wide = 1 if lw >= lh else 0
                            right_tall = 1 if rh >= rw else 0

                            # Prefer pure rotations over reflections by ordering
                            # all_transforms() with rotations first.
                            left_order = all_transforms(left_comp["piece"]).index(left_t)
                            right_order = all_transforms(right_comp["piece"]).index(right_t)
                            top_order = all_transforms(top_piece).index(top_t)

                            # Prefer the filler's empty bbox to touch top/right edges.
                            empty_crop = crop_mask(empty)
                            e_r0, e_r1 = np.where(empty.any(axis=1))[0][[0, -1]]
                            e_c0, e_c1 = np.where(empty.any(axis=0))[0][[0, -1]]
                            empty_touch = 1 if (e_r0 == 0 and e_c1 == N - 1) else 0

                            score = (
                                filler_match,
                                left_wide,
                                right_tall,
                                -top_order,
                                -left_order,
                                -right_order,
                                empty_touch,
                                -left_idx,
                                -right_idx
                            )

                            if best_score is None or score > best_score:
                                out = np.zeros((N, N), dtype=int)
                                out[0:th, 0:tw] = top_t
                                out[lr:lr + lh, lc:lc + lw] = left_t
                                out[rr:rr + rh, rc:rc + rw] = right_t

                                # Fill remaining cells.
                                # If the filler piece exactly matches the empty
                                # shape, use its colors; otherwise use dominant color.
                                used_filler_colors = False
                                for ft in all_transforms(filler_comp["piece"]):
                                    fh, fw = ft.shape
                                    if ft.shape == empty_crop.shape and np.array_equal(ft > 0, empty_crop):
                                        # Place it aligned to the empty bounding box.
                                        r0, c0 = e_r0, e_c0
                                        out[r0:r0 + fh, c0:c0 + fw] = ft
                                        used_filler_colors = True
                                        break

                                if not used_filler_colors:
                                    fill_color = dominant_color(filler_comp["piece"])
                                    out[empty] = fill_color

                                best_score = score
                                best_output = out

    if best_output is None:
        # Extremely defensive fallback: simply pack all non-zero cells row-major
        # into an N x N square.  This is not expected to be needed for valid
        # instances, but avoids returning None.
        flat = grid[grid > 0].astype(int).flatten()
        best_output = np.zeros(N * N, dtype=int)
        if flat.size <= N * N:
            best_output[:flat.size] = flat
        best_output = best_output.reshape(N, N)

    return best_output