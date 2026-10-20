import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the compact hidden object from the 20x20 scattered input.

    The examples show the same generative pattern:
    - the final object is formed by 3 or 4 colored layers
    - one color forms the outer/bounding layer
    - one color forms the inner layer
    - sometimes a center marker color is present
    - the input contains translated / flipped / rotated copies of those layers
      plus isolated decoy pixels

    This solver therefore:
    1. Removes low-frequency decoy pixels.
    2. Finds the main rectangular reference frame from the largest meaningful layer.
    3. Reconstructs the output in that frame.
    4. For each non-outer color, tries the identity and simple dihedral transforms
       (horizontal flip, vertical flip, 180 rotation) and keeps the placement that
       maximizes agreement with the observed pixels of that color.
    5. Returns the compact reconstructed frame.
    """

    def crop_bbox(arr):
        mask = arr != 0
        if not mask.any():
            return np.zeros((1, 1), dtype=int)
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        return arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()

    if grid.size == 0:
        return np.zeros_like(grid)

    colors = [int(c) for c in np.unique(grid) if c != 0]
    if not colors:
        return np.zeros_like(grid)

    # Remove obvious isolated decoys: keep pixels of each color that belong
    # to the densest main cluster by using connected components (4-connectivity).
    def keep_main_component(mask):
        if not mask.any():
            return mask
        h, w = mask.shape
        lab = -1 - np.zeros_like(mask, dtype=int)
        cur = -1
        components = []
        for i in range(h):
            for j in range(w):
                if mask[i, j] and lab[i, j] == -1:
                    cur += 1
                    stack = [(i, j)]
                    lab[i, j] = cur
                    comp = []
                    while stack:
                        y, x = stack.pop()
                        comp.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if mask[ny, nx] and lab[ny, nx] == -1:
                                    lab[ny, nx] = cur
                                    stack.append((ny, nx))
                    components.append(comp)

        if not components:
            return mask

        # Choose the component with maximum size; ties by lower bbox area.
        best = max(
            components,
            key=lambda comp: (
                len(comp),
                -((max(y for y, x in comp) - min(y for y, x in comp)) *
                  (max(x for y, x in comp) - min(x for y, x in comp)))
            )
        )

        new_mask = np.zeros_like(mask, dtype=bool)
        for y, x in best:
            new_mask[y, x] = True
        return new_mask

    cleaned = grid.copy()
    for c in colors:
        mask = grid == c
        cleaned[mask & ~keep_main_component(mask)] = 0

    # If cleaning removed everything for a color, fall back to original.
    if not (cleaned != 0).any():
        cleaned = grid.copy()

    # Determine reference frame from the most extensive non-zero structure.
    # Prefer the color whose cleaned pixels have the largest bounding box.
    best_ref = None
    best_score = -1
    for c in colors:
        mask = cleaned == c
        if not mask.any():
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        h = rows[-1] - rows[0] + 1
        w = cols[-1] - cols[0] + 1
        score = h * w * 10 + mask.sum()
        if score > best_score:
            best_score = score
            best_ref = (c, rows[0], rows[-1], cols[0], cols[-1])

    if best_ref is None:
        return np.zeros_like(grid)

    ref_color, r0, r1, c0, c1 = best_ref
    H = r1 - r0 + 1
    W = c1 - c0 + 1

    out = np.zeros((H, W), dtype=int)

    # Place the reference color in identity orientation.
    out[(cleaned == ref_color) & np.pad(
        np.ones_like(cleaned, dtype=bool), ((r0, H - r1), (c0, W - c1))
    )] = ref_color

    # Simpler direct crop: extract reference color crop.
    out = (cleaned[r0:r1 + 1, c0:c1 + 1] == ref_color).astype(int) * ref_color

    def place_color(color, orientation):
        """Try to place a color layer relative to the reference crop."""
        mask = cleaned == color
        if not mask.any():
            return out, 0

        crop = np.zeros((H, W), dtype=bool)
        for y, x in np.argwhere(mask):
            ry, rx = y - r0, x - c0
            if 0 <= ry < H and 0 <= rx < W:
                crop[ry, rx] = True

        if not crop.any():
            return out, 0

        # We want to know if current out already has this color correctly.
        # For reconstruction, we place the best orientation matching the crop.
        # Since input may contain transformed copy, use symmetry of the observed crop:
        # choose orientation that maximizes overlap after transforming the canonical
        # unknown layer. We infer the canonical layer by trying to "undo" transforms
        # and checking consistency with the reference crop boundaries.

        candidates = {
            "id": crop,
            "hflip": crop[:, ::-1],
            "vflip": crop[::-1, :],
            "rot180": crop[::-1, ::-1],
        }

        # Score each orientation by how well it aligns with the observed color
        # after applying the inverse transform.
        best_score = -1
        best_layer = None
        for name, cand in candidates.items():
            # A layer is plausible if its non-zero cells all fit inside the
            # reference frame and do not conflict with the reference color.
            conflict = np.sum(cand & (out != 0) & (out != color))
            if conflict > 0:
                continue
            # Prefer orientations with a tighter bbox centered inside the frame.
            rows = np.where(cand.any(axis=1))[0]
            cols = np.where(cand.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                continue
            ch = rows[-1] - rows[0] + 1
            cw = cols[-1] - cols[0] + 1
            cy = (rows[0] + rows[-1]) / 2.0
            cx = (cols[0] + cols[-1]) / 2.0
            center_dist = abs(cy - H / 2) + abs(cx - W / 2)
            score = cand.sum() * 100 - ch * cw * 0.1 - center_dist * 10
            if score > best_score:
                best_score = score
                best_layer = cand

        if best_layer is None:
            return out, 0

        new_out = out.copy()
        new_out[best_layer] = color
        return new_out, best_score

    # Process non-reference colors from outside-in by frequency:
    # smaller/rarer layers are usually inner markers.
    color_counts = {c: int((cleaned == c).sum()) for c in colors}
    order = sorted([c for c in colors if c != ref_color], key=lambda c: color_counts[c], reverse=False)

    for c in order:
        out, _ = place_color(c, None)

    # Crop final output to its tight non-zero bounding box.
    out = crop_bbox(out)

    # If the result is too small (all decoys removed), fallback to the reference crop.
    if out.size < max(1, (H * W) // 4):
        out = crop_bbox((cleaned == ref_color).astype(int) * ref_color)

    return out