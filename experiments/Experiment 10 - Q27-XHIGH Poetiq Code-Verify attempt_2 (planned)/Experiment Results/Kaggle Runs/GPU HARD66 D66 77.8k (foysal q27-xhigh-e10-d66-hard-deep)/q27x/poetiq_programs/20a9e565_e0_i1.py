import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    1. Color 5 marks two 2x2 corner brackets. They define the output size:
       - top-left bracket gives the top/left corner
       - bottom-right bracket gives the bottom/right corner
       Output height/width are the inclusive bounding box of all 5s.

    2. Select the source object using the top row of that 5-bounding box:
       - ignore 0 and 5
       - choose the leftmost non-5 object intersecting the top 5-row
       - if several candidates are equally close, prefer the one whose
         horizontal span matches the output width, then the leftmost one

    3. Build the output by taking the selected object's rows:
       - if the object is shorter than the output height, tile it vertically
       - if the object is wider than the output width, take the leftmost
         output-width window in each row
       - if the object is narrower than the output width, horizontally
         stretch it while keeping 1-pixel strokes as thin as possible

    The examples are handled by this behavior.
    """
    if grid is None or grid.size == 0:
        return np.zeros_like(grid)

    h, w = grid.shape

    # --- 1. Find the rectangle defined by color 5 -------------------------
    ys, xs = np.where(grid == 5)
    if ys.size == 0:
        return np.zeros((1, 1), dtype=grid.dtype)

    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    out_h = int(bottom - top + 1)
    out_w = int(right - left + 1)

    # --- 2. Find candidate source objects in the top 5-row ----------------
    top_row = grid[top]

    # connected components of all non-0/non-5 pixels
    visited = np.zeros((h, w), dtype=bool)
    candidates = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or grid[r, c] == 0 or grid[r, c] == 5:
                continue

            # BFS component
            color = grid[r, c]
            comp_cells = []
            stack = [(r, c)]
            visited[r, c] = True

            while stack:
                cr, cc = stack.pop()
                comp_cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not visited[nr, nc] and grid[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            if not comp_cells:
                continue

            comp = np.array(comp_cells)
            rmin, rmax = comp[:, 0].min(), comp[:, 0].max()
            cmin, cmax = comp[:, 1].min(), comp[:, 1].max()

            # candidate must intersect the top 5-row
            if not (rmin <= top <= rmax):
                continue

            # score: prefer leftmost, then width matching out_w
            width = int(cmax - cmin + 1)
            score = (0 if width == out_w else 1, int(cmin), -width)
            candidates.append((score, cmin, cmax, rmin, rmax, color))

    if not candidates:
        return np.zeros((out_h, out_w), dtype=grid.dtype)

    # pick best candidate
    candidates.sort(key=lambda x: x[0])
    _, cmin, cmax, rmin, rmax, main_color = candidates[0]
    src_w = int(cmax - cmin + 1)
    src_h = int(rmax - rmin + 1)

    # Extract the selected object's bounding box.
    obj = grid[rmin:rmax + 1, cmin:cmax + 1].copy()
    obj[(obj == 0) | (obj == 5)] = 0

    # If the object contains multiple colors, keep all non-zero colors.
    # If it is single-color, keep only that color.
    non_zero_colors = np.unique(obj[obj != 0])
    if len(non_zero_colors) == 1:
        keep = non_zero_colors[0]
        obj[obj != keep] = 0
    else:
        # In multi-color cases, keep all non-zero pixels.
        pass

    # --- 3. Build source rows to the required width -----------------------
    def stretch_row(row, target_w):
        """
        Stretch a 1D row to target width while preserving 1-pixel strokes.
        This is sufficient for the small patterns in this task.
        """
        row = np.asarray(row)
        src_w = len(row)
        if src_w == target_w:
            return row.copy()

        # For simple line/C shapes, map non-zero intervals to target row.
        out = np.zeros(target_w, dtype=row.dtype)

        # Find contiguous non-zero intervals.
        nz = row != 0
        i = 0
        while i < src_w:
            if not nz[i]:
                i += 1
                continue
            j = i
            while j + 1 < src_w and nz[j + 1]:
                j += 1

            # Interval [i, j] inclusive in source.
            # Map to target preserving at least 1 pixel.
            ti = int(round(i * (target_w - 1) / max(src_w - 1, 1)))
            tj = int(round(j * (target_w - 1) / max(src_w - 1, 1)))
            ti = max(0, min(target_w - 1, ti))
            tj = max(0, min(target_w - 1, tj))
            if tj < ti:
                ti, tj = tj, ti

            # Fill the mapped interval with the first non-zero color.
            color = row[i]
            out[ti:tj + 1] = color
            i = j + 1

        return out

    src_rows = []
    for r in range(src_h):
        row = obj[r]
        if src_w <= out_w:
            src_rows.append(stretch_row(row, out_w))
        else:
            # Take the leftmost out_w pixels of the object row.
            # This works for the examples where the selected object is
            # wider than the requested crop.
            src_rows.append(row[:out_w].copy())

    # --- 4. Tile vertically to output height ------------------------------
    result = np.zeros((out_h, out_w), dtype=grid.dtype)
    for i in range(out_h):
        result[i] = src_rows[i % src_h]

    return result