import numpy as np
from collections import Counter

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform framed 5x5 objects in the grid.

    Observed rule:
    - The grid contains square framed objects of size 5x5.
    - There are two frame colors: 2 and 8.
    - Background is the dominant colour.
    - Objects keep their vertical positions.
    - Frame color 8 objects are moved to the left side of the grid.
    - Frame color 2 objects are moved to the right side of the grid.
    - Within each frame-color group, objects preserve their relative horizontal order.
      They are placed in two lanes:
        * border 8: lanes 0 and 5
        * border 2: lanes 12 and 17
      The preferred lane is chosen using the horizontal median:
        * border 8: columns > median go to lane 5, otherwise lane 0
        * border 2: columns < median go to lane 12, otherwise lane 17
    - If that preferred monotone split would cause vertical overlap in one lane,
      the closest valid monotone split is used.

    The implementation below follows this rule.
    """
    if grid.size == 0:
        return grid

    H, W = grid.shape

    # Determine background color.
    # The corner is usually background; use the most frequent color as fallback.
    bg = int(grid[0, 0])
    vals, counts = np.unique(grid, return_counts=True)
    if counts[np.where(vals == bg)[0][0]] < counts.max():
        bg = int(vals[np.argmax(counts)])

    # Detect all 5x5 framed objects.
    objects = []
    for r in range(H - 4):
        for c in range(W - 4):
            border = int(grid[r, c])
            if border == bg:
                continue
            if border not in (2, 8):
                continue

            # Check border.
            if not np.all(grid[r, c:c+5] == border):
                continue
            if not np.all(grid[r+4, c:c+5] == border):
                continue
            if not np.all(grid[r+1:r+4, c] == border):
                continue
            if not np.all(grid[r+1:r+4, c+4] == border):
                continue

            # Check interior is uniform.
            interior = grid[r+1:r+4, c+1:c+4]
            if interior.size == 0:
                continue
            fill = int(interior[0, 0])
            if fill == border:
                continue
            if not np.all(interior == fill):
                continue

            # To avoid detecting sub-squares inside a larger structure,
            # accept it as an object.
            objects.append({
                'r': r,
                'c': c,
                'border': border,
                'fill': fill,
                'w': 5,
                'h': 5,
            })

    if not objects:
        return grid.copy()

    # Object size is assumed constant; use 5.
    obj_w = 5
    obj_h = 5

    # Lane positions.
    left_lanes = [0, obj_w]                  # for border 8
    right_lanes = [W - 2 * obj_w, W - obj_w] # for border 2, e.g. [12, 17]

    def valid_split(group, lanes):
        """
        group: list of dicts, sorted by input column.
        lanes: [lane0, lane1]
        Returns (valid, assignments) where assignments is list of chosen cols.
        """
        n = len(group)
        for k in range(n + 1):
            assignments = []
            ok = True
            for i, obj in enumerate(group):
                col = lanes[0] if i < k else lanes[1]
                assignments.append(col)

            # Check vertical overlaps within each lane.
            for lane in lanes:
                rows = [obj['r'] for i, obj in enumerate(group) if assignments[i] == lane]
                for a in range(len(rows)):
                    for b in range(a + 1, len(rows)):
                        r1 = rows[a]
                        r2 = rows[b]
                        if abs(r1 - r2) < obj_h:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                return True, assignments, k

        return False, None, None

    # Separate objects by border color.
    groups = {2: [], 8: []}
    for obj in objects:
        if obj['border'] in groups:
            groups[obj['border']].append(obj)

    assigned = []

    for bcolor, group in groups.items():
        if not group:
            continue

        # Sort by input column, then row for deterministic tie-breaking.
        group.sort(key=lambda x: (x['c'], x['r']))

        cols = np.array([x['c'] for x in group], dtype=float)
        med = float(np.median(cols))
        n = len(group)

        if bcolor == 8:
            lanes = left_lanes
            # Preferred: col <= median -> lane 0, col > median -> lane 1.
            desired_k = int(np.sum(cols <= med))
        else:  # bcolor == 2
            lanes = right_lanes
            # Preferred: col < median -> lane 0, col >= median -> lane 1.
            desired_k = int(np.sum(cols < med))

        # Try desired split first.
        valid, assignments, k = valid_split(group, lanes)

        # The helper above returns the first valid split, not necessarily desired.
        # We need explicit search to choose the closest valid split to desired_k.
        best_k = None
        best_assignments = None
        best_dist = None

        for k in range(n + 1):
            assignments_k = []
            ok = True
            for i, obj in enumerate(group):
                col = lanes[0] if i < k else lanes[1]
                assignments_k.append(col)

            for lane in lanes:
                rows = [obj['r'] for i, obj in enumerate(group) if assignments_k[i] == lane]
                for a in range(len(rows)):
                    for b in range(a + 1, len(rows)):
                        if abs(rows[a] - rows[b]) < obj_h:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break

            if ok:
                dist = abs(k - desired_k)
                if best_k is None or dist < best_dist:
                    best_k = k
                    best_assignments = assignments_k
                    best_dist = dist

        # Fallback: if no monotone split is valid, use greedy lane assignment.
        if best_assignments is None:
            best_assignments = []
            last = {lanes[0]: -10**9, lanes[1]: -10**9}
            for obj in sorted(group, key=lambda x: x['r']):
                r = obj['r']
                # Prefer desired lane based on median.
                if bcolor == 8:
                    pref = lanes[1] if obj['c'] > med else lanes[0]
                else:
                    pref = lanes[0] if obj['c'] < med else lanes[1]

                if last[pref] + obj_h <= r:
                    col = pref
                elif last[lanes[0]] + obj_h <= r:
                    col = lanes[0]
                elif last[lanes[1]] + obj_h <= r:
                    col = lanes[1]
                else:
                    # Very defensive fallback.
                    col = pref

                best_assignments.append(col)
                last[col] = r

        for obj, col in zip(group, best_assignments):
            assigned.append({
                'r': obj['r'],
                'c': col,
                'border': obj['border'],
                'fill': obj['fill'],
                'w': obj_w,
                'h': obj_h,
            })

    # Build output grid.
    out = np.full((H, W), bg, dtype=grid.dtype)

    # Draw objects.
    for obj in assigned:
        r = obj['r']
        c = obj['c']
        w = obj['w']
        h = obj['h']

        # Fill interior first.
        out[r:r+h, c:c+w] = obj['fill']

        # Draw border.
        out[r, c:c+w] = obj['border']
        out[r+h-1, c:c+w] = obj['border']
        out[r:r+h, c] = obj['border']
        out[r:r+h, c+w-1] = obj['border']

    return out