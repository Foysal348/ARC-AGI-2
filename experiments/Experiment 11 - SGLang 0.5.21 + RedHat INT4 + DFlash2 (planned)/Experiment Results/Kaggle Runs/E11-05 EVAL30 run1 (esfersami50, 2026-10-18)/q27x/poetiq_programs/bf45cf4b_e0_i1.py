import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform rule:
    1. Find the background color.
    2. Identify the tile object as the multi-color non-background component.
    3. Treat all remaining non-background pixels as the mask pattern.
    4. Crop the tight bounding boxes of the tile and the mask.
    5. Build an output grid whose size is (mask_h * tile_h, mask_w * tile_w).
       For each pixel in the mask bounding box:
       - if it is part of the mask, paste the tile there;
       - otherwise fill with the background color.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    # Background is the most frequent color, preferring corners on ties.
    vals, counts = np.unique(grid, return_counts=True)
    max_count = counts.max()
    candidates = vals[counts == max_count]
    corners = {
        int(grid[0, 0]),
        int(grid[0, -1]),
        int(grid[-1, 0]),
        int(grid[-1, -1]),
    }
    bg = int(next((c for c in candidates if c in corners), candidates[0]))

    nonbg = grid != bg
    if not nonbg.any():
        return grid.copy()

    # 4-connected component labeling for non-background pixels.
    def label_components(mask):
        h, w = mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        label = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] and labels[r, c] == 0:
                    label += 1
                    labels[r, c] = label
                    stack = [(r, c)]

                    while stack:
                        cr, cc = stack.pop()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < h
                                and 0 <= nc < w
                                and mask[nr, nc]
                                and labels[nr, nc] == 0
                            ):
                                labels[nr, nc] = label
                                stack.append((nr, nc))

        return labels, label

    labels, n_labels = label_components(nonbg)
    if n_labels == 0:
        return grid.copy()

    # Collect component metadata.
    comp_info = []
    for lab in range(1, n_labels + 1):
        ys, xs = np.nonzero(labels == lab)
        if ys.size == 0:
            continue

        min_r, max_r = int(ys.min()), int(ys.max())
        min_c, max_c = int(xs.min()), int(xs.max())
        area = int(ys.size)
        bbox_area = (max_r - min_r + 1) * (max_c - min_c + 1)
        colors = tuple(int(c) for c in np.unique(grid[ys, xs]))

        comp_info.append({
            "label": lab,
            "area": area,
            "bbox_area": bbox_area,
            "bbox": (min_r, max_r, min_c, max_c),
            "colors": colors,
            "density": area / bbox_area,
        })

    if not comp_info:
        return grid.copy()

    # The tile is the multi-color component. If all components are monochrome,
    # fall back to the densest/largest rectangular-looking object.
    multi = [ci for ci in comp_info if len(ci["colors"]) > 1]
    if multi:
        stamp = max(
            multi,
            key=lambda ci: (ci["bbox_area"], len(ci["colors"]), ci["area"]),
        )
    else:
        stamp = max(
            comp_info,
            key=lambda ci: (ci["density"], ci["bbox_area"], ci["area"]),
        )

    # Mask pixels are all non-background pixels outside the stamp component.
    mask_pixels = nonbg & (labels != stamp["label"])

    if not mask_pixels.any():
        # Safety fallback: use the largest monochromatic non-stamp component.
        mono = [
            ci for ci in comp_info
            if len(ci["colors"]) == 1 and ci["label"] != stamp["label"]
        ]
        if mono:
            best = max(mono, key=lambda ci: (ci["area"], ci["bbox_area"]))
            mask_pixels = labels == best["label"]
        else:
            return grid.copy()

    # Crop tight bounding boxes.
    ys, xs = np.nonzero(mask_pixels)
    mask_min_r, mask_max_r = int(ys.min()), int(ys.max())
    mask_min_c, mask_max_c = int(xs.min()), int(xs.max())
    mask_h = mask_max_r - mask_min_r + 1
    mask_w = mask_max_c - mask_min_c + 1

    s_min_r, s_max_r, s_min_c, s_max_c = stamp["bbox"]
    tile = grid[s_min_r:s_max_r + 1, s_min_c:s_max_c + 1].copy()
    tile_h, tile_w = tile.shape

    # Expand the mask into the output grid.
    out = np.full((mask_h * tile_h, mask_w * tile_w), bg, dtype=grid.dtype)

    for i in range(mask_h):
        for j in range(mask_w):
            if mask_pixels[mask_min_r + i, mask_min_c + j]:
                out[i * tile_h:(i + 1) * tile_h,
                    j * tile_w:(j + 1) * tile_w] = tile

    return out