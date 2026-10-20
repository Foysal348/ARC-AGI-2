import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by moving some 2-pieces into container cavities.

    Rule inferred from examples:
    - The background color is the dominant border/background color.
    - Non-background, non-2, non-0 pixels form container components.
    - Each connected component of 2s is considered.
    - A 2-component may move in one of the four cardinal directions if:
        * all leading-edge cells see the same container distance,
        * the touched container pixel has the same container immediately behind it
          (so it is an internal lip, not an outer boundary),
        * the final shifted component lies inside a cavity of that container
          (between two lateral walls perpendicular to the motion),
        * the swept path does not cross any container pixel.
    - Moved components are translated until adjacent to the internal wall.
    - All cells traversed by a moved component, except its final pixels,
      become 0.
    - Every original 2 is erased from its original position.
    """
    arr = np.asarray(grid, dtype=int)
    if arr.size == 0:
        return arr

    H, W = arr.shape

    # Determine background color. Prefer global mode, but fall back to border mode.
    max_val = int(arr.max()) if arr.size else 0
    minlength = max(10, max_val + 1)

    all_counts = np.bincount(arr.ravel(), minlength=minlength)
    bg = int(np.argmax(all_counts))

    border = np.concatenate((arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]))
    border_counts = np.bincount(border.astype(int), minlength=minlength)
    if border_counts[bg] == 0 and border_counts.max() > 0:
        bg = int(np.argmax(border_counts))

    result = arr.copy()

    # If there are no 2s, nothing changes.
    if not np.any(arr == 2):
        return result

    # Container pixels: everything that is not background, not 2, and not trail 0.
    container_mask = (arr != bg) & (arr != 2) & (arr != 0)

    # Label connected container components (4-connectivity).
    comp = np.full((H, W), -1, dtype=int)
    bboxes = []
    dirs4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if container_mask[r, c] and comp[r, c] == -1:
                cid = len(bboxes)
                stack = [(r, c)]
                comp[r, c] = cid
                min_r = max_r = r
                min_c = max_c = c

                while stack:
                    cr, cc = stack.pop()
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)

                    for dr, dc in dirs4:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if container_mask[nr, nc] and comp[nr, nc] == -1:
                                comp[nr, nc] = cid
                                stack.append((nr, nc))

                bboxes.append((min_r, max_r, min_c, max_c))

    def is_inside_cavity(r: int, c: int, dr: int, dc: int, wid: int) -> bool:
        """
        Check that (r,c) is inside a cavity of component wid:
        - for vertical motion, there must be container pixels left and right
          in the same row;
        - for horizontal motion, there must be container pixels above and below
          in the same column.
        """
        if wid < 0 or wid >= len(bboxes):
            return False

        min_r, max_r, min_c, max_c = bboxes[wid]
        if not (min_r <= r <= max_r and min_c <= c <= max_c):
            return False

        if dr != 0:
            # Moving up/down: need lateral walls left/right.
            left = bool(np.any(comp[r, min_c:c] == wid))
            right = bool(np.any(comp[r, c + 1:max_c + 1] == wid))
            return left and right
        else:
            # Moving left/right: need lateral walls above/below.
            up = bool(np.any(comp[min_r:r, c] == wid))
            down = bool(np.any(comp[r + 1:max_r + 1, c] == wid))
            return up and down

    # Find connected components of 2s.
    visited2 = np.zeros((H, W), dtype=bool)
    moves = []

    for r in range(H):
        for c in range(W):
            if arr[r, c] == 2 and not visited2[r, c]:
                stack = [(r, c)]
                visited2[r, c] = True
                cells = []

                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))

                    for dr, dc in dirs4:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if arr[nr, nc] == 2 and not visited2[nr, nc]:
                                visited2[nr, nc] = True
                                stack.append((nr, nc))

                comp_set = set(cells)
                best_dir = None
                best_shift = None

                # Try all four cardinal directions.
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    valid = True
                    shifts = []
                    wall_pixels = []

                    # Leading-edge cells are those with no own cell immediately
                    # in the movement direction.
                    for (cr, cc) in cells:
                        front_r, front_c = cr + dr, cc + dc
                        if (front_r, front_c) in comp_set:
                            continue

                        nr, nc = cr + dr, cc + dc
                        steps = 0
                        found = False

                        while 0 <= nr < H and 0 <= nc < W:
                            steps += 1

                            if comp[nr, nc] != -1:
                                # Distance needed to become adjacent to this wall.
                                shifts.append(steps - 1)
                                wall_pixels.append((nr, nc))
                                found = True
                                break

                            nr += dr
                            nc += dc

                        if not found:
                            valid = False
                            break

                    if not valid or not shifts:
                        continue

                    # All leading cells must hit a wall at the same depth.
                    if len(set(shifts)) != 1:
                        continue

                    shift = shifts[0]
                    if shift <= 0:
                        continue

                    # All touched walls must belong to the same container component.
                    wids = {comp[wr, wc] for (wr, wc) in wall_pixels}
                    if len(wids) != 1:
                        continue
                    wid = wids.pop()

                    # The touched wall must have the same container immediately
                    # behind it (internal lip, not outer boundary).
                    ok = True
                    for (wr, wc) in wall_pixels:
                        br, bc = wr + dr, wc + dc
                        if not (0 <= br < H and 0 <= bc < W) or comp[br, bc] != wid:
                            ok = False
                            break

                    if not ok:
                        continue

                    # The whole shifted component must lie inside the cavity.
                    for (cr, cc) in cells:
                        tr, tc = cr + dr * shift, cc + dc * shift
                        if not (0 <= tr < H and 0 <= tc < W):
                            ok = False
                            break
                        if comp[tr, tc] != -1:
                            ok = False
                            break
                        if not is_inside_cavity(tr, tc, dr, dc, wid):
                            ok = False
                            break

                    if not ok:
                        continue

                    # Ensure the swept path does not cross any container pixel.
                    path_ok = True
                    for s in range(shift + 1):
                        for (cr, cc) in cells:
                            nr, nc = cr + dr * s, cc + dc * s
                            if not (0 <= nr < H and 0 <= nc < W):
                                path_ok = False
                                break
                            if comp[nr, nc] != -1:
                                path_ok = False
                                break
                        if not path_ok:
                            break

                    if path_ok:
                        if best_shift is None or shift < best_shift:
                            best_shift = shift
                            best_dir = (dr, dc)

                if best_dir is not None:
                    moves.append((cells, best_dir[0], best_dir[1], best_shift))

    # Erase all original 2s.
    result[arr == 2] = bg

    if moves:
        # First lay down all trails (0s).
        trail = np.zeros((H, W), dtype=bool)
        for cells, dr, dc, shift in moves:
            for r, c in cells:
                for s in range(shift + 1):
                    nr, nc = r + dr * s, c + dc * s
                    if 0 <= nr < H and 0 <= nc < W and comp[nr, nc] == -1:
                        trail[nr, nc] = True

        result[trail] = 0

        # Then place moved 2s at their final positions.
        for cells, dr, dc, shift in moves:
            for r, c in cells:
                nr, nc = r + dr * shift, c + dc * shift
                if 0 <= nr < H and 0 <= nc < W and comp[nr, nc] == -1:
                    result[nr, nc] = 2

    return result