import numpy as np
from collections import deque


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Denoise colored objects by restoring their dominant mirror symmetry.

    For every same-colored 8-connected component:
      1. Try all vertical and horizontal mirror axes.
      2. For each axis, keep only pixels that either lie on the axis or have
         their mirrored pixel present in the original component.
      3. Score candidate symmetric cores using:
           - number of kept pixels
           - how symmetric the core is in the perpendicular direction
           - penalty for 2x2 solid blocks (often caused by noise/thickening)
           - penalty for the core becoming disconnected
      4. Keep the highest-scoring symmetric core.

    This removes added "bump" pixels while preserving the intended symmetric
    object.
    """
    if grid.size == 0:
        return grid

    g = grid.astype(np.int32, copy=True)
    h, w = g.shape

    # Background is the most common border color.
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    if border.size == 0:
        bg = 0
    else:
        vals, counts = np.unique(border, return_counts=True)
        bg = int(vals[np.argmax(counts)])

    out = np.full((h, w), bg, dtype=grid.dtype)

    def count_2x2(mask: np.ndarray) -> int:
        """Count fully occupied 2x2 blocks in a boolean mask."""
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            return 0
        return int(np.sum(
            mask[:-1, :-1] &
            mask[1:, :-1] &
            mask[:-1, 1:] &
            mask[1:, 1:]
        ))

    def count_components(coords):
        """Count 8-connected components among a list of coordinates."""
        if not coords:
            return 0

        remaining = set(coords)
        cnt = 0

        while remaining:
            start = remaining.pop()
            cnt += 1
            q = deque([start])

            while q:
                r, c = q.popleft()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in remaining:
                            remaining.remove((nr, nc))
                            q.append((nr, nc))

        return cnt

    def sym_count(mask: np.ndarray, coords, orient: str) -> int:
        """
        For a mask, return the maximum number of pixels that are symmetric
        about some axis in the given orientation ('V' or 'H').
        """
        if not coords:
            return 0

        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        best = 0

        if orient == 'V':
            # axis2 = 2 * axis_coordinate
            # even axis2: axis through a column
            # odd axis2: axis between columns
            for axis2 in range(2 * cmin, 2 * cmax + 1):
                half = (axis2 % 2 == 1)
                center = axis2 // 2
                cnt = 0

                if half:
                    for r, c in coords:
                        mc = axis2 - c
                        if cmin <= mc <= cmax and mask[r, mc]:
                            cnt += 1
                else:
                    for r, c in coords:
                        if c == center:
                            cnt += 1
                        else:
                            mc = axis2 - c
                            if cmin <= mc <= cmax and mask[r, mc]:
                                cnt += 1

                if cnt > best:
                    best = cnt

        else:
            for axis2 in range(2 * rmin, 2 * rmax + 1):
                half = (axis2 % 2 == 1)
                center = axis2 // 2
                cnt = 0

                if half:
                    for r, c in coords:
                        mr = axis2 - r
                        if rmin <= mr <= rmax and mask[mr, c]:
                            cnt += 1
                else:
                    for r, c in coords:
                        if r == center:
                            cnt += 1
                        else:
                            mr = axis2 - r
                            if rmin <= mr <= rmax and mask[mr, c]:
                                cnt += 1

                if cnt > best:
                    best = cnt

        return best

    def process_component(comp_mask: np.ndarray, coords, color: int):
        """Restore one object component and write it into out."""
        if len(coords) <= 2:
            out[comp_mask] = color
            return

        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        best_key = None
        best_core = None

        # Try vertical first so vertical wins ties.
        for orient in ('V', 'H'):
            if orient == 'V':
                axis_start, axis_end = 2 * cmin, 2 * cmax
                center2 = cmin + cmax
            else:
                axis_start, axis_end = 2 * rmin, 2 * rmax
                center2 = rmin + rmax

            for axis2 in range(axis_start, axis_end + 1):
                half = (axis2 % 2 == 1)
                center = axis2 // 2

                core_coords = []
                on_axis = 0

                crmin, crmax = h, -1
                ccmin, ccmax = w, -1

                if orient == 'V':
                    for r, c in coords:
                        keep = False

                        if not half and c == center:
                            keep = True
                            on_axis += 1
                        else:
                            mc = axis2 - c
                            if cmin <= mc <= cmax and comp_mask[r, mc]:
                                keep = True

                        if keep:
                            core_coords.append((r, c))
                            if r < crmin:
                                crmin = r
                            if r > crmax:
                                crmax = r
                            if c < ccmin:
                                ccmin = c
                            if c > ccmax:
                                ccmax = c

                else:
                    for r, c in coords:
                        keep = False

                        if not half and r == center:
                            keep = True
                            on_axis += 1
                        else:
                            mr = axis2 - r
                            if rmin <= mr <= rmax and comp_mask[mr, c]:
                                keep = True

                        if keep:
                            core_coords.append((r, c))
                            if r < crmin:
                                crmin = r
                            if r > crmax:
                                crmax = r
                            if c < ccmin:
                                ccmin = c
                            if c > ccmax:
                                ccmax = c

                R = len(core_coords)
                if R == 0:
                    continue

                core_mask = np.zeros((h, w), dtype=bool)
                rows = [r for r, _ in core_coords]
                cols = [c for _, c in core_coords]
                core_mask[rows, cols] = True

                B = count_2x2(core_mask)
                S = sym_count(core_mask, core_coords, 'H' if orient == 'V' else 'V')
                N = count_components(core_coords)

                score = R + S - 3 * B - 2 * (N - 1)

                dist = abs(axis2 - center2)
                bbox_area = (crmax - crmin + 1) * (ccmax - ccmin + 1)
                orient_pref = 1 if orient == 'V' else 0

                # Higher is better.
                key = (score, orient_pref, -dist, on_axis, -bbox_area)

                if best_key is None or key > best_key:
                    best_key = key
                    best_core = core_mask

        if best_core is None:
            out[comp_mask] = color
        else:
            out[best_core] = color

    # Process each non-background color separately, using 8-connected components.
    for color in [int(c) for c in np.unique(g) if c != bg]:
        color_mask = (g == color)
        unvisited = color_mask.copy()

        for i in range(h):
            for j in range(w):
                if not unvisited[i, j]:
                    continue

                comp_mask = np.zeros((h, w), dtype=bool)
                q = deque([(i, j)])
                unvisited[i, j] = False
                comp_mask[i, j] = True
                coords = []

                while q:
                    r, c = q.popleft()
                    coords.append((r, c))

                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and unvisited[nr, nc]:
                                unvisited[nr, nc] = False
                                comp_mask[nr, nc] = True
                                q.append((nr, nc))

                process_component(comp_mask, coords, int(color))

    return out