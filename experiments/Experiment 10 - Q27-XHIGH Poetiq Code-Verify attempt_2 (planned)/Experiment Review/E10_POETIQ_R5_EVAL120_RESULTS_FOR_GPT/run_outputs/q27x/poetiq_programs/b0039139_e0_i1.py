import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Split the input into panels by lines of 1s.

    - Use the first sparse object panel as the source pattern.
    - Use the first two solid panels as the output colors:
        first solid color -> object cells
        second solid color -> background / gap cells
    - Crop the source pattern to its bounding box.
    - If panels are arranged horizontally, repeat the pattern horizontally
      as many times as its height, separated by one column, and remove the
      final separator column.
    - If panels are arranged vertically, repeat the pattern vertically
      as many times as its width, separated by one row, and remove the
      final separator row.
    """
    arr = np.array(grid, dtype=int)
    if arr.size == 0:
        return np.zeros_like(arr)

    H, W = arr.shape

    # Full separator rows/columns of color 1.
    sep_rows = [r for r in range(H) if np.all(arr[r] == 1)]
    sep_cols = [c for c in range(W) if np.all(arr[:, c] == 1)]

    if not sep_rows and not sep_cols:
        return arr.copy()

    def split_panels(horizontal: bool):
        """Split along separator columns if horizontal, else along rows."""
        panels = []
        if horizontal:
            prev = 0
            for c in sep_cols:
                if c > prev:
                    panels.append(arr[:, prev:c])
                prev = c + 1
            if prev < W:
                panels.append(arr[:, prev:])
        else:
            prev = 0
            for r in sep_rows:
                if r > prev:
                    panels.append(arr[prev:r, :])
                prev = r + 1
            if prev < H:
                panels.append(arr[prev:, :])
        return panels

    def classify_panels(panels):
        """
        Return:
          masks: bounding masks for sparse pattern panels
          solids: colors from fully solid panels, in panel order
        """
        masks = []
        solids = []

        for panel in panels:
            if panel.size == 0:
                continue

            unique = np.unique(panel)
            nonsep = [int(c) for c in unique if c not in (0, 1)]

            # Fully solid panel.
            if len(unique) == 1:
                color = int(unique[0])
                if color not in (0, 1):
                    solids.append(color)
                continue

            # Sparse pattern panel: contains background 0 and at least one object color.
            if 0 in unique and nonsep:
                mask = (panel != 0) & (panel != 1)
                if np.any(mask):
                    masks.append(mask)
                continue

            # Fallback for unexpected panels.
            if nonsep:
                vals, counts = np.unique(panel, return_counts=True)
                dom = int(vals[np.argmax(counts)])
                if counts.max() == panel.size:
                    solids.append(dom)
                else:
                    mask = (panel == dom)
                    if np.any(mask):
                        masks.append(mask)

        return masks, solids

    # Decide orientation. Normally only one direction has separators.
    if sep_cols and not sep_rows:
        horizontal = True
    elif sep_rows and not sep_cols:
        horizontal = False
    else:
        # If both exist, choose the direction that better parses the panels.
        h_panels = split_panels(True)
        v_panels = split_panels(False)
        h_masks, h_solids = classify_panels(h_panels)
        v_masks, v_solids = classify_panels(v_panels)

        h_score = (len(h_masks) > 0) + (len(h_solids) >= 2) + 2 * len(sep_cols)
        v_score = (len(v_masks) > 0) + (len(v_solids) >= 2) + 2 * len(sep_rows)
        horizontal = h_score >= v_score

    panels = split_panels(horizontal)
    masks, solid_colors = classify_panels(panels)

    if not masks:
        return np.zeros_like(arr)

    # Fallback colors if solid detection is incomplete.
    if len(solid_colors) < 2:
        vals, _ = np.unique(arr[arr != 0], return_counts=True)
        candidates = [int(v) for v in vals if int(v) != 1]
        if len(candidates) >= 2:
            solid_colors = candidates[:2]
        elif candidates:
            solid_colors = candidates + [0]
        else:
            solid_colors = [1, 0]

    color_obj = int(solid_colors[0])
    color_bg = int(solid_colors[1])

    # Use the first sparse object panel.
    mask = masks[0]

    # Crop to the object's bounding box.
    row_has = np.any(mask, axis=1)
    col_has = np.any(mask, axis=0)
    if not np.any(row_has) or not np.any(col_has):
        return np.zeros_like(arr)

    r_idx = np.where(row_has)[0]
    c_idx = np.where(col_has)[0]
    mask = mask[r_idx[0]:r_idx[-1] + 1, c_idx[0]:c_idx[-1] + 1]

    h, w = mask.shape

    if horizontal:
        # Repeat horizontally as many times as the object's height.
        repeats = h
        tile = np.pad(mask.astype(int), ((0, 0), (0, 1)), mode="constant", constant_values=0)
        if repeats > 1:
            tiled = np.concatenate([tile] * repeats, axis=1)
        else:
            tiled = tile
        out = tiled[:, :-1]  # remove final gap column
    else:
        # Repeat vertically as many times as the object's width.
        repeats = w
        tile = np.pad(mask.astype(int), ((0, 1), (0, 0)), mode="constant", constant_values=0)
        if repeats > 1:
            tiled = np.concatenate([tile] * repeats, axis=0)
        else:
            tiled = tile
        out = tiled[:-1, :]  # remove final gap row

    return np.where(out.astype(bool), color_obj, color_bg).astype(int)