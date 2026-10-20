import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the single canonical display by:

    1. Treating the most common color as background.
    2. Treating the most common non-background color as the frame/wall color.
    3. Grouping the remaining non-background colors into visual tiles.
    4. Stacking those tiles vertically in their original top-to-bottom order.
    5. Centering them horizontally and overlaying their contents.
    6. Drawing the reconstructed frame as the 8-connected dilation of all
       non-background content, cropped to the final bounding box.

    This implementation is deliberately robust for the observed ARC style:
    scattered framed colored pieces are assembled into one compact output
    display.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, dtype=np.uint8, copy=True)
    h, w = grid.shape

    # ---------------------------------------------------------------
    # 1. Identify background and frame colors.
    # ---------------------------------------------------------------
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) == 1:
        return grid.copy()

    bg_color = colors[np.argmax(counts)]

    # If the background color is not unique-ish, still use the most frequent.
    non_bg_counts = []
    for c in colors:
        if c == bg_color:
            continue
        non_bg_counts.append((c, counts[np.argmax(colors == c)]))

    if not non_bg_counts:
        return grid.copy()

    # Frame color is the most frequent non-background color.
    frame_color = max(non_bg_counts, key=lambda x: x[1])[0]

    # Content colors are all non-background, non-frame colors.
    content_mask = np.zeros_like(grid, dtype=bool)
    content_colors = set()
    for c in colors:
        if c != bg_color and c != frame_color:
            content_mask[grid == c] = True
            content_colors.add(c)

    # If there is no content, return a cropped frame component or the grid.
    if not content_colors:
        # Fall back to cropping the largest frame component.
        frame_mask = (grid == frame_color)
        if not frame_mask.any():
            return grid.copy()

        # Simple connected component labeling for 4-connectivity.
        from collections import deque

        best = None
        seen = np.zeros_like(frame_mask, dtype=bool)
        for r in range(h):
            for c in range(w):
                if frame_mask[r, c] and not seen[r, c]:
                    q = deque([(r, c)])
                    seen[r, c] = True
                    comp = [(r, c)]
                    while q:
                        y, x = q.popleft()
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and frame_mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                                comp.append((ny, nx))
                    if best is None or len(comp) > best[0]:
                        best = (len(comp), comp)
        if best is None:
            return grid.copy()
        ys = [p[0] for p in best[1]]
        xs = [p[1] for p in best[1]]
        out = np.full((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1), bg_color, dtype=np.uint8)
        for y, x in best[1]:
            out[y - min(ys), x - min(xs)] = frame_color
        return out

    # ---------------------------------------------------------------
    # 2. Group content pixels into tiles.
    #
    # Content pixels are grouped if their bounding boxes overlap after a
    # small expansion. This merges horizontally adjacent pieces such as
    # left/right side pieces and central pieces belonging to one tile.
    # ---------------------------------------------------------------
    content_coords = np.argwhere(content_mask)
    if len(content_coords) == 0:
        return grid.copy()

    # Connected components of content pixels using 8-connectivity.
    from collections import deque

    seen_content = np.zeros_like(grid, dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if content_mask[r, c] and not seen_content[r, c]:
                q = deque([(r, c)])
                seen_content[r, c] = True
                comp = [(r, c)]
                while q:
                    y, x = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and content_mask[ny, nx] and not seen_content[ny, nx]:
                                seen_content[ny, nx] = True
                                q.append((ny, nx))
                                comp.append((ny, nx))
                ys = [p[0] for p in comp]
                xs = [p[1] for p in comp]
                components.append({
                    "pixels": comp,
                    "y0": min(ys),
                    "y1": max(ys),
                    "x0": min(xs),
                    "x1": max(xs),
                })

    # Merge components whose expanded boxes overlap.
    merged = []
    used = [False] * len(components)
    EPS = 2  # pixels of allowed gap for horizontal/vertical merging

    for i in range(len(components)):
        if used[i]:
            continue
        used[i] = True
        y0, y1 = components[i]["y0"], components[i]["y1"]
        x0, x1 = components[i]["x0"], components[i]["x1"]
        changed = True

        while changed:
            changed = False
            for j in range(len(components)):
                if used[j]:
                    continue
                cy0, cy1 = components[j]["y0"], components[j]["y1"]
                cx0, cx1 = components[j]["x0"], components[j]["x1"]
                # Expand both boxes by EPS and test overlap.
                if (y0 - EPS <= cx1 and x1 + EPS >= cx0 and
                    y1 + EPS >= cy0 and x0 - EPS <= cx1):
                    used[j] = True
                    y0 = min(y0, cy0)
                    y1 = max(y1, cy1)
                    x0 = min(x0, cx0)
                    x1 = max(x1, cx1)
                    changed = True

        pixels = []
        for j in range(len(components)):
            if used[j]:
                pixels.extend(components[j]["pixels"])

        merged.append({
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
            "pixels": pixels,
        })

    # Remove tiny noise components.
    tiles = []
    for t in merged:
        if len(t["pixels"]) < 2:
            continue
        tiles.append(t)

    if not tiles:
        return grid.copy()

    # Sort tiles by vertical position, then horizontal position.
    tiles.sort(key=lambda t: (t["y0"], t["x0"]))

    # ---------------------------------------------------------------
    # 3. Build the stacked content canvas.
    #
    # Each tile is cropped to its content bounding box.  The tiles are
    # stacked vertically with a one-pixel vertical overlap, which matches
    # the observed examples where adjacent framed tiles share a frame row.
    # ---------------------------------------------------------------
    tile_patterns = []
    max_w = 0
    total_h = 0

    for idx, t in enumerate(tiles):
        sub = grid[t["y0"]:t["y1"] + 1, t["x0"]:t["x1"] + 1].copy()
        # Clear background and frame pixels inside the tile; keep content colors.
        sub[(sub == bg_color)] = 0
        sub[(sub == frame_color)] = 0
        tile_patterns.append(sub)
        max_w = max(max_w, sub.shape[1])
        total_h += sub.shape[0]

    # If there are multiple tiles, reduce total height by one row per overlap.
    if len(tiles) > 1:
        total_h -= (len(tiles) - 1)

    # If total height is too small, use at least the largest tile height.
    total_h = max(total_h, max(tp.shape[0] for tp in tile_patterns))

    content_canvas = np.zeros((total_h, max_w), dtype=np.uint8)
    y = 0
    for tp in tile_patterns:
        th, tw = tp.shape
        x = (max_w - tw) // 2
        # Overlay content; non-zero overwrites zero.
        content_canvas[y:y + th, x:x + tw] = np.maximum(
            content_canvas[y:y + th, x:x + tw],
            tp
        )
        # Next tile starts one row below the current bottom, with 1-row overlap.
        if y + th >= total_h:
            break
        y += max(1, th - 1)

    # ---------------------------------------------------------------
    # 4. Reconstruct the frame.
    #
    # The output frame is generated as the 8-neighbor dilation of the
    # non-background content pixels.  This creates the thin frame visible
    # in the examples without needing to recover every fragmented wall
    # component directly from the input.
    # ---------------------------------------------------------------
    content_nonzero = (content_canvas > 0)

    frame_canvas = np.zeros_like(content_canvas, dtype=bool)
    if content_nonzero.any():
        from scipy import ndimage
        # If scipy is unavailable, fallback to manual dilation.
        try:
            frame_canvas = ndimage.binary_dilation(content_nonzero, iterations=1)
        except Exception:
            frame_canvas = content_nonzero.copy()
            for r in range(total_h):
                for c in range(max_w):
                    if not frame_canvas[r, c]:
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = r + dy, c + dx
                                if 0 <= ny < total_h and 0 <= nx < max_w and content_nonzero[ny, nx]:
                                    frame_canvas[r, c] = True
                                    break
                            else:
                                continue
                            break

    # Put content on top of frame.
    out = np.where(frame_canvas, frame_color, 0).astype(np.uint8)
    out[content_canvas > 0] = content_canvas[content_canvas > 0]

    # ---------------------------------------------------------------
    # 5. Crop to the final bounding box and restore background color.
    # ---------------------------------------------------------------
    # The background outside the frame should be the original background.
    out[(out == 0)] = bg_color

    rows = np.any((out != bg_color) | frame_canvas, axis=1)
    cols = np.any((out != bg_color) | frame_canvas, axis=0)

    if not rows.any() or not cols.any():
        return out

    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]

    return out[r0:r1 + 1, c0:c1 + 1].copy()