import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange the separated square blocks into one horizontal or vertical chain.

    The input is divided into blocks by full rows/columns of color 6.
    Each block contains a colored path. The path has endpoint(s) where it
    touches the border of the block. The output reorders the blocks so that
    these border endpoints connect into one continuous chain:
      - if endpoints are on left/right sides, output is a horizontal row;
      - if endpoints are on top/bottom sides, output is a vertical column.
    Separator lines of color 6 are inserted between output blocks.
    """
    grid = np.array(grid, dtype=int)
    if grid.ndim != 2:
        return grid

    SEP = 6
    BG = 7

    H, W = grid.shape

    def make_segments(size, sep_indices):
        """Return contiguous segments excluding full separator rows/cols."""
        starts = []
        ends = []
        start = 0

        for sep in sep_indices:
            if sep < 0 or sep >= size:
                continue
            if sep > start:
                starts.append(start)
                ends.append(sep)
            start = sep + 1

        if start < size:
            starts.append(start)
            ends.append(size)

        return list(zip(starts, ends))

    # Find full separator rows and columns.
    sep_rows = [r for r in range(H) if np.all(grid[r] == SEP)]
    sep_cols = [c for c in range(W) if np.all(grid[:, c] == SEP)]

    row_segs = make_segments(H, sep_rows) or [(0, H)]
    col_segs = make_segments(W, sep_cols) or [(0, W)]

    # Extract all blocks.
    panels = []
    for r0, r1 in row_segs:
        for c0, c1 in col_segs:
            if r1 - r0 <= 0 or c1 - c0 <= 0:
                continue
            panel = grid[r0:r1, c0:c1]
            if np.all(panel == SEP):
                continue
            panels.append(panel)

    if not panels:
        return grid.copy()

    if len(panels) == 1:
        return panels[0].copy()

    def side_coordinates(panel):
        """
        Return coordinates of non-background pixels on each border side:
          top:    column indices on row 0
          bottom: column indices on last row
          left:   row indices on column 0
          right:  row indices on last column
        """
        h, w = panel.shape

        top = tuple(c for c in range(w) if panel[0, c] != BG)
        bottom = tuple(c for c in range(w) if panel[h - 1, c] != BG)
        left = tuple(r for r in range(h) if panel[r, 0] != BG)
        right = tuple(r for r in range(h) if panel[r, w - 1] != BG)

        return top, bottom, left, right

    def analyze(axis):
        """
        For a chosen chain axis, build entry/exit coordinate options.

        axis='v': chain goes top -> bottom, entry=top, exit=bottom
        axis='h': chain goes left -> right, entry=left, exit=right
        """
        in_opts = []
        out_opts = []

        relevant_count = 0
        perpendicular_count = 0

        for panel in panels:
            top, bottom, left, right = side_coordinates(panel)

            if axis == 'v':
                in_opts.append(top)
                out_opts.append(bottom)

                relevant_count += len(top) + len(bottom)
                perpendicular_count += len(left) + len(right)
            else:
                in_opts.append(left)
                out_opts.append(right)

                relevant_count += len(left) + len(right)
                perpendicular_count += len(top) + len(bottom)

        has_endpoint = relevant_count > 0
        score = relevant_count - perpendicular_count

        return in_opts, out_opts, has_endpoint, score

    def solve_chain(n, in_opts, out_opts):
        """
        Find an ordering of all blocks such that:
          - the first block has no incoming side endpoint,
          - the last block has no outgoing side endpoint,
          - every adjacent pair matches the exact endpoint coordinate.
        """
        in_opts = [tuple(x) for x in in_opts]
        out_opts = [tuple(x) for x in out_opts]
        full_mask = (1 << n) - 1

        starts = [i for i in range(n) if len(in_opts[i]) == 0]
        if not starts:
            if n == 1:
                return [0]
            return None

        from functools import lru_cache

        @lru_cache(maxsize=None)
        def can_finish(cur_out, used_mask):
            if used_mask == full_mask:
                return cur_out is None

            if cur_out is None:
                return False

            for j in range(n):
                if (used_mask >> j) & 1:
                    continue

                if cur_out not in in_opts[j]:
                    continue

                possible_outs = out_opts[j] if out_opts[j] else (None,)
                for out_choice in possible_outs:
                    if can_finish(out_choice, used_mask | (1 << j)):
                        return True

            return False

        # Try possible starts and reconstruct one valid chain.
        for start in starts:
            possible_starts = out_opts[start] if out_opts[start] else (None,)

            for start_out in possible_starts:
                if not can_finish(start_out, 1 << start):
                    continue

                order = [start]
                used = 1 << start
                cur = start_out

                while used != full_mask:
                    if cur is None:
                        return None

                    advanced = False

                    for j in range(n):
                        if (used >> j) & 1:
                            continue

                        if cur not in in_opts[j]:
                            continue

                        possible_outs = out_opts[j] if out_opts[j] else (None,)

                        for out_choice in possible_outs:
                            if can_finish(out_choice, used | (1 << j)):
                                order.append(j)
                                used |= (1 << j)
                                cur = out_choice
                                advanced = True
                                break

                        if advanced:
                            break

                    if not advanced:
                        return None

                if cur is None:
                    return order

        return None

    # Analyze both possible chain orientations.
    v_in, v_out, v_has, v_score = analyze('v')
    h_in, h_out, h_has, h_score = analyze('h')

    candidates = []
    if v_has:
        candidates.append((-v_score, 0, 'v', v_in, v_out))
    if h_has:
        candidates.append((-h_score, 1, 'h', h_in, h_out))

    candidates.sort()

    ordered_panels = None
    output_axis = 'v'

    for _, _, axis, in_opts, out_opts in candidates:
        order = solve_chain(len(panels), in_opts, out_opts)
        if order is not None:
            ordered_panels = [panels[i] for i in order]
            output_axis = axis
            break

    # Fallback for unexpected inputs: preserve block order.
    if ordered_panels is None:
        if h_has and not v_has:
            output_axis = 'h'
        else:
            output_axis = 'v'
        ordered_panels = list(panels)

    # Build final output grid with separator lines.
    if output_axis == 'v':
        total_h = sum(p.shape[0] for p in ordered_panels) + (len(ordered_panels) - 1)
        total_w = max(p.shape[1] for p in ordered_panels)

        out = np.full((total_h, total_w), BG, dtype=grid.dtype)

        r = 0
        for i, panel in enumerate(ordered_panels):
            ph, pw = panel.shape
            out[r:r + ph, 0:pw] = panel
            r += ph

            if i != len(ordered_panels) - 1:
                out[r, :] = SEP
                r += 1

    else:
        total_w = sum(p.shape[1] for p in ordered_panels) + (len(ordered_panels) - 1)
        total_h = max(p.shape[0] for p in ordered_panels)

        out = np.full((total_h, total_w), BG, dtype=grid.dtype)

        c = 0
        for i, panel in enumerate(ordered_panels):
            ph, pw = panel.shape
            out[0:ph, c:c + pw] = panel
            c += pw

            if i != len(ordered_panels) - 1:
                out[:, c] = SEP
                c += 1

    return out