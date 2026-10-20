import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input grid according to the following rule:

    1. Find the vertical separator column made of 4s.
    2. The output has the input's height and width equal to the number of
       columns to the left of that separator.
    3. Copy the marker 5 into the output at the same row and at the same
       column offset from the separator.
    4. Read the 3x3 symbols on the left side track-by-track:
       left track top-to-bottom, then right track top-to-bottom.
    5. Convert each symbol color to a line segment:
         1 -> horizontal length 3, traversed left-to-right
         2 -> horizontal length 2, traversed right-to-left
         3 -> horizontal length 4, traversed right-to-left
         6 -> vertical length 2
    6. Starting just below the 5, place the segments vertically so that the
       entry end of each new segment touches the exit point of the previous one.
    """
    grid = np.asarray(grid)

    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return np.zeros_like(grid)

    # ------------------------------------------------------------------
    # Find the vertical separator column of 4s.
    # ------------------------------------------------------------------
    sep_col = -1
    for c in range(W):
        if np.all(grid[:, c] == 4):
            sep_col = c
            break

    if sep_col == -1:
        counts = np.sum(grid == 4, axis=0)
        if counts.size:
            best = int(np.argmax(counts))
            if counts[best] >= max(1, H // 2):
                sep_col = best

    if sep_col <= 0 or sep_col >= W:
        return np.zeros((H, max(0, W - 1)), dtype=grid.dtype)

    out_w = sep_col
    out = np.zeros((H, out_w), dtype=grid.dtype)

    # ------------------------------------------------------------------
    # Locate the marker 5.
    # ------------------------------------------------------------------
    has_marker = False
    marker_row = 0
    anchor = out_w // 2

    marker_cells = np.argwhere(grid == 5)
    if marker_cells.size:
        has_marker = True
        r, c = marker_cells[0]
        r = int(r)
        c = int(c)
        marker_row = r

        if c > sep_col:
            anchor = c - sep_col - 1
        elif c < sep_col:
            anchor = c
        else:
            anchor = out_w // 2

        anchor = int(np.clip(anchor, 0, out_w - 1))
        if 0 <= marker_row < H:
            out[marker_row, anchor] = 5

    # ------------------------------------------------------------------
    # Parse the left-side symbols into an ordered color sequence.
    # ------------------------------------------------------------------
    left = grid[:, :sep_col]
    obj_mask = (left != 0) & (left != 4) & (left != 5)

    if not np.any(obj_mask):
        return out

    # Find vertical tracks (slots) as contiguous columns containing symbols.
    col_has = np.any(obj_mask, axis=0)
    tracks = []
    c = 0
    while c < out_w:
        if col_has[c]:
            start = c
            while c < out_w and col_has[c]:
                c += 1
            tracks.append((start, c))
        else:
            c += 1

    if not tracks:
        return out

    sequence = []

    # Process each track left-to-right, and within each track top-to-bottom.
    for track_start, track_end in tracks:
        slot_mask = obj_mask[:, track_start:track_end]
        row_has = np.any(slot_mask, axis=1)

        r = 0
        while r < H:
            if not row_has[r]:
                r += 1
                continue

            start_r = r
            while r < H and row_has[r]:
                r += 1

            # Split vertical components into 3-row symbol blocks.
            rr = start_r
            while rr < H and row_has[rr]:
                end_r = min(rr + 3, H)
                block_mask = slot_mask[rr:end_r, :]
                vals = left[rr:end_r, track_start:track_end][block_mask]

                if vals.size:
                    vals = vals[vals > 0]
                    vals = vals[(vals != 4) & (vals != 5)]
                    if vals.size:
                        unique_vals, counts = np.unique(vals, return_counts=True)
                        color = int(unique_vals[np.argmax(counts)])
                        sequence.append(color)

                rr = end_r

    # ------------------------------------------------------------------
    # Draw the chain of line segments below the marker.
    # ------------------------------------------------------------------
    # color -> (type, length, entry_side, exit_side)
    tokens = {
        1: ("H", 3, "left", "right"),
        2: ("H", 2, "right", "left"),
        3: ("H", 4, "right", "left"),
        6: ("V", 2, "same", "same"),
    }

    next_row = marker_row + 1 if has_marker else 0
    if next_row < 0:
        next_row = 0

    cur = anchor

    for color in sequence:
        if next_row >= H:
            break

        # Fallback for unexpected colors: draw a single pixel.
        if color not in tokens:
            if 0 <= cur < out_w:
                out[next_row, cur] = color
            next_row += 1
            continue

        ttype, length, entry, exit = tokens[color]

        if ttype == "V":
            col = int(np.clip(cur, 0, out_w - 1))
            top = next_row
            bottom = top + length - 1

            if bottom >= H:
                out[top:H, col] = color
                next_row = H
                cur = col
                break

            out[top:bottom + 1, col] = color
            next_row = bottom + 1
            cur = col

        else:
            if length == 1:
                start = end = int(np.clip(cur, 0, out_w - 1))
            elif entry == "right":
                end = cur
                start = end - length + 1
            else:  # entry == "left"
                start = cur
                end = start + length - 1

            # Keep the segment within the output width.
            if start < 0:
                shift = -start
                start += shift
                end += shift

            if end >= out_w:
                shift = end - (out_w - 1)
                start -= shift
                end -= shift

            start = int(max(start, 0))
            end = int(min(end, out_w - 1))

            if start > end:
                start = 0
                end = min(length, out_w) - 1

            out[next_row, start:end + 1] = color

            if exit == "left":
                cur = start
            elif exit == "right":
                cur = end
            else:
                cur = int(np.clip(cur, start, end))

            next_row += 1

    return out