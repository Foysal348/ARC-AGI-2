import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the following rule:

    1. Find a 2x2 background gap whose four neighboring 2-cell sides are
       occupied by four different non-background colors:
           top, bottom, left, right markers.
    2. Keep those four marker pairs in place.
    3. For each marker color, take all other pixels of that color
       (the "remote" shape), crop them to a binary pattern, and choose one of
       its 8 symmetries so that the side facing the marker contains exactly
       one adjacent pair of 1s.
       - Prefer the pair centered on the facing side.
       - Then prefer the smallest symmetry cost.
    4. Place that remote shape directly outside the marker, aligning the
       chosen pair with the marker.
    5. All other colors are removed.

    This matches the examples: small marker pairs define a cross around a
    2x2 hole, and remote same-color pieces are folded outward from the markers.
    """
    if grid.size == 0:
        return grid

    # Background is the most frequent color (always 8 in the given tasks).
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    H, W = grid.shape
    out = np.full((H, W), bg, dtype=grid.dtype)

    def is_bg(x):
        return int(x) == bg

    def valid_nonbg_pair(row, col, length):
        """Return color if a 2-cell segment is fully non-background."""
        if length == 2:
            vals = grid[row, col:col+2] if isinstance(row, (int, np.integer)) else None
        return None  # placeholder, not used

    # Find the central 2x2 gap with four adjacent marker pairs.
    gap = None
    marker_colors = None

    for r in range(1, H - 2):
        for c in range(1, W - 2):
            # 2x2 gap must be background.
            if not np.all(grid[r:r+2, c:c+2] == bg):
                continue

            # Top marker: two cells above gap.
            top_vals = grid[r-1, c:c+2]
            if top_vals[0] == bg or top_vals[0] != top_vals[1]:
                continue

            # Bottom marker: two cells below gap.
            bot_vals = grid[r+2, c:c+2]
            if bot_vals[0] == bg or bot_vals[0] != bot_vals[1]:
                continue

            # Left marker: two cells left of gap.
            left_vals = grid[r:r+2, c-1]
            if left_vals[0] == bg or left_vals[0] != left_vals[1]:
                continue

            # Right marker: two cells right of gap.
            right_vals = grid[r:r+2, c+1]
            if right_vals[0] == bg or right_vals[0] != right_vals[1]:
                continue

            colors = [int(top_vals[0]), int(bot_vals[0]),
                      int(left_vals[0]), int(right_vals[0])]

            # In all examples, the four marker colors are distinct.
            if len(set(colors)) != 4:
                continue

            # Prefer a gap that actually has remote pixels for at least one color.
            gap = (r, c)
            marker_colors = colors
            break

        if gap is not None:
            break

    # If no cross is found, return the original grid unchanged.
    if gap is None:
        return grid

    gr, gc = gap
    top_color, bot_color, left_color, right_color = marker_colors

    def marker_pixels(direction, color):
        if direction == "top":
            return [(gr - 1, gc), (gr - 1, gc + 1)]
        if direction == "bottom":
            return [(gr + 2, gc), (gr + 2, gc + 1)]
        if direction == "left":
            return [(gr, gc - 1), (gr + 1, gc - 1)]
        if direction == "right":
            return [(gr, gc + 1), (gr + 1, gc + 1)]
        return []

    def get_remote_matrix(color, marker_pos):
        mask = (grid == color)
        for (r, c) in marker_pos:
            mask[r, c] = False

        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None

        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        sub = (grid[r0:r1+1, c0:c1+1] == color).astype(np.uint8)
        return sub

    def symmetries(mat):
        """Return unique 8 binary symmetries with a small cost tie-breaker."""
        res = {}

        def add(arr, cost):
            key = tuple(map(tuple, arr.tolist()))
            if key not in res or cost < res[key][1]:
                res[key] = (np.array(arr, dtype=np.uint8), cost)

        m = mat
        add(m, 0)
        add(np.rot90(m, 1), 1)
        add(np.rot90(m, 2), 2)
        add(np.rot90(m, 3), 1)

        add(m[:, ::-1], 1)
        add(m[::-1, :], 1)
        add(m[::-1, ::-1], 2)

        add(np.rot90(m, 1)[:, ::-1], 2)
        add(np.rot90(m, 2)[:, ::-1], 2)
        add(np.rot90(m, 3)[:, ::-1], 2)

        return list(res.values())

    def facing_vector(t, direction):
        if direction == "top":
            return t[-1, :]
        if direction == "bottom":
            return t[0, :]
        if direction == "left":
            return t[:, -1]
        if direction == "right":
            return t[:, 0]
        return None

    def block_start(vec):
        L = len(vec)
        if L < 2:
            return None
        if L == 2:
            return 0 if vec[0] == 1 and vec[1] == 1 else None
        ones = np.nonzero(vec == 1)[0]
        if len(ones) != 2:
            return None
        if ones[1] == ones[0] + 1:
            return int(ones[0])
        return None

    def choose_remote_orientation(mat, direction):
        """Choose a symmetry of mat with a valid facing side."""
        candidates = []

        for t, cost in symmetries(mat):
            v = facing_vector(t, direction)
            if v is None:
                continue

            L = len(v)
            s = block_start(v)

            if s is not None:
                # Exact ideal case: exactly one adjacent pair on the facing side.
                center_dist = abs(s - (L - 2) / 2.0)
                candidates.append((0, center_dist, cost, t, s))
            else:
                # Fallback: score every possible 2-cell alignment.
                best = None
                for s in range(L - 1):
                    cover = int(v[s]) + int(v[s + 1])
                    extra = int(np.sum(v))
                    center_dist = abs(s - (L - 2) / 2.0)
                    # Prefer covering both marker cells, then fewer extras,
                    # then centered, then cheaper symmetry.
                    score = (2 - cover, extra, center_dist, cost)
                    if best is None or score < best[0]:
                        best = (score, t, s)
                if best is not None:
                    candidates.append((1, 0, cost, best[1], best[2]))

        if not candidates:
            return None, None

        # Primary: exact ideal; secondary: centered; tertiary: symmetry cost.
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, _, _, best_t, best_s = candidates[0]
        return best_t, best_s

    def place_remote(direction, color, marker_pos):
        remote = get_remote_matrix(color, marker_pos)

        # Always keep the marker itself.
        for (r, c) in marker_pos:
            out[r, c] = color

        if remote is None:
            return

        t, s = choose_remote_orientation(remote, direction)
        if t is None or s is None:
            return

        h, w = t.shape

        if direction == "top":
            # Remote is above the top marker.
            marker_row = gr - 1
            top_row = marker_row - h
            left_col = gc - s
            out[top_row:top_row+h, left_col:left_col+w] = np.where(t > 0, color, out[top_row:top_row+h, left_col:left_col+w])

        elif direction == "bottom":
            # Remote is below the bottom marker.
            marker_row = gr + 2
            top_row = marker_row + 1
            left_col = gc - s
            out[top_row:top_row+h, left_col:left_col+w] = np.where(t > 0, color, out[top_row:top_row+h, left_col:left_col+w])

        elif direction == "left":
            # Remote is left of the left marker.
            marker_col = gc - 1
            top_row = gr - s
            left_col = marker_col - w
            out[top_row:top_row+h, left_col:left_col+w] = np.where(t > 0, color, out[top_row:top_row+h, left_col:left_col+w])

        elif direction == "right":
            # Remote is right of the right marker.
            marker_col = gc + 1
            top_row = gr - s
            left_col = marker_col + 1
            out[top_row:top_row+h, left_col:left_col+w] = np.where(t > 0, color, out[top_row:top_row+h, left_col:left_col+w])

    # Place each colored arm.
    place_remote("top", top_color, marker_pixels("top", top_color))
    place_remote("bottom", bot_color, marker_pixels("bottom", bot_color))
    place_remote("left", left_color, marker_pixels("left", left_color))
    place_remote("right", right_color, marker_pixels("right", right_color))

    return out