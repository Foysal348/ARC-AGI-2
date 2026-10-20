import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Find a straight separator line. The side containing colored pixels is the
    source side. In each row/column, maximal contiguous non-background runs are
    read outward from the separator. A run of length L and color C projects a
    periodic pattern onto the opposite side: cells at 1-indexed distances
    1, 1+L, 1+2L, ... from the separator become C. If several runs project to
    the same cell, the run nearest the separator wins.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2:
        return grid.copy()

    out = grid.copy()
    H, W = grid.shape

    if H < 3 or W < 3:
        return out

    # Background is taken as the most frequent color.
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def line_score(axis):
        """Return (score, index) for the best internal line in the given axis."""
        best_score = 0
        best_idx = -1

        if axis == 'v':
            for c in range(1, W - 1):
                col = grid[:, c]
                cv, cc = np.unique(col, return_counts=True)
                maxc = 0
                for v, cnt in zip(cv, cc):
                    if v != bg and cnt > maxc:
                        maxc = cnt
                if maxc > best_score:
                    best_score = maxc
                    best_idx = c
        else:
            for r in range(1, H - 1):
                row = grid[r, :]
                rv, rc = np.unique(row, return_counts=True)
                maxc = 0
                for v, cnt in zip(rv, rc):
                    if v != bg and cnt > maxc:
                        maxc = cnt
                if maxc > best_score:
                    best_score = maxc
                    best_idx = r

        return best_score, best_idx

    v_score, v_idx = line_score('v')
    h_score, h_idx = line_score('h')

    # Require a reasonably long line to consider it a separator.
    threshold = max(2, min(H, W) // 3)
    if max(v_score, h_score) < threshold:
        return out

    # Choose the best candidate line. Also require that there is at least
    # some non-background content on one of the two sides.
    candidates = []

    if v_idx != -1 and v_score >= threshold:
        side_total = int(
            np.count_nonzero(grid[:, :v_idx] != bg) +
            np.count_nonzero(grid[:, v_idx + 1:] != bg)
        )
        if side_total > 0:
            candidates.append((v_score, 'v', v_idx))

    if h_idx != -1 and h_score >= threshold:
        side_total = int(
            np.count_nonzero(grid[:h_idx, :] != bg) +
            np.count_nonzero(grid[h_idx + 1:, :] != bg)
        )
        if side_total > 0:
            candidates.append((h_score, 'h', h_idx))

    if not candidates:
        return out

    # Prefer the longer line; on ties prefer vertical because all examples are vertical.
    candidates.sort(key=lambda x: (-x[0], 0 if x[1] == 'v' else 1))
    axis, sep = candidates[0][1], candidates[0][2]

    def parse_blocks(seq):
        """
        Parse maximal contiguous non-background runs in a 1D sequence read
        outward from the separator.

        Returns a list of (color, width, nearest_distance_from_separator).
        """
        seq = np.asarray(seq)
        n = len(seq)
        blocks = []
        i = 0

        while i < n:
            if seq[i] == bg:
                i += 1
                continue

            j = i + 1
            while j < n and seq[j] == seq[i]:
                j += 1

            color = int(seq[i])
            width = j - i
            nearest = i + 1  # 1-indexed distance of first cell of this block
            blocks.append((color, width, nearest))

            i = j

        return blocks

    def make_pattern(seq, length):
        """
        Build the projected 1D pattern of given length from source sequence seq.
        Farther blocks are painted first, so nearer blocks override overlaps.
        """
        if length <= 0:
            return np.array([], dtype=grid.dtype)

        res = np.full(length, bg, dtype=grid.dtype)
        blocks = parse_blocks(seq)

        # Paint from farthest to nearest.
        for color, width, nearest in sorted(blocks, key=lambda b: b[2], reverse=True):
            # Indices 0, width, 2*width correspond to distances 1, 1+width, ...
            res[::width] = color

        return res

    if axis == 'v':
        c = sep

        left_total = int(np.count_nonzero(grid[:, :c] != bg))
        right_total = int(np.count_nonzero(grid[:, c + 1:] != bg))
        global_source_left = left_total >= right_total

        for r in range(H):
            left_cnt = int(np.count_nonzero(grid[r, :c] != bg))
            right_cnt = int(np.count_nonzero(grid[r, c + 1:] != bg))

            if left_cnt == 0 and right_cnt == 0:
                continue

            if left_cnt == 0:
                source_left = False
            elif right_cnt == 0:
                source_left = True
            else:
                source_left = global_source_left

            if source_left:
                target_len = W - c - 1
                if target_len <= 0:
                    continue
                source_seq = grid[r, :c][::-1]  # nearest to farthest
                pattern = make_pattern(source_seq, target_len)
                out[r, c + 1:] = pattern
            else:
                target_len = c
                if target_len <= 0:
                    continue
                source_seq = grid[r, c + 1:]    # nearest to farthest
                pattern = make_pattern(source_seq, target_len)
                out[r, :c] = pattern[::-1]

    else:
        r = sep

        above_total = int(np.count_nonzero(grid[:r, :] != bg))
        below_total = int(np.count_nonzero(grid[r + 1:, :] != bg))
        global_source_above = above_total >= below_total

        for c in range(W):
            above_cnt = int(np.count_nonzero(grid[:r, c] != bg))
            below_cnt = int(np.count_nonzero(grid[r + 1:, c] != bg))

            if above_cnt == 0 and below_cnt == 0:
                continue

            if above_cnt == 0:
                source_above = False
            elif below_cnt == 0:
                source_above = True
            else:
                source_above = global_source_above

            if source_above:
                target_len = H - r - 1
                if target_len <= 0:
                    continue
                source_seq = grid[:r, c][::-1]  # nearest to farthest
                pattern = make_pattern(source_seq, target_len)
                out[r + 1:, c] = pattern
            else:
                target_len = r
                if target_len <= 0:
                    continue
                source_seq = grid[r + 1:, c]    # nearest to farthest
                pattern = make_pattern(source_seq, target_len)
                out[:r, c] = pattern[::-1]

    return out