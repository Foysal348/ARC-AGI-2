import numpy as np
from scipy import ndimage


def _components(grid: np.ndarray):
    """
    Return list of components.
    Each component is dict:
      - mask: binary array for that component
      - color: original representative color
      - r0, r1, c0, c1: bounding box
      - r, c: centroid
      - top, right, bottom, left: normalized corner positions in [0, 1]
    """
    labeled, n = ndimage.label(grid != 0)
    components = []

    for idx in range(1, n + 1):
        mask = labeled == idx
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue

        r0, r1 = ys.min(), ys.max()
        c0, c1 = xs.min(), xs.max()

        # Representative color: most common non-zero color in this component
        vals = grid[mask]
        unique, counts = np.unique(vals, return_counts=True)
        color = int(unique[np.argmax(counts)])

        h = r1 - r0 + 1
        w = c1 - c0 + 1

        centroid_r = np.mean(ys)
        centroid_c = np.mean(xs)

        # Normalized position within its own bounding box
        top = centroid_r / max(h - 1, 1)
        bottom = 1.0 - top
        left = centroid_c / max(w - 1, 1)
        right = 1.0 - left

        components.append({
            "mask": mask,
            "color": color,
            "r0": int(r0),
            "r1": int(r1),
            "c0": int(c0),
            "c1": int(c1),
            "r": centroid_r,
            "c": centroid_c,
            "top": top,
            "right": right,
            "bottom": bottom,
            "left": left,
        })

    return components


def _find_palette_2x2(grid: np.ndarray):
    """
    Find the 2x2 block containing four distinct non-zero colors.
    Returns:
      palette: 2x2 int array
      mask: boolean mask of that 2x2 region
    """
    H, W = grid.shape
    for r in range(H - 1):
        for c in range(W - 1):
            block = grid[r:r + 2, c:c + 2]
            vals = block.flatten().tolist()
            if all(v != 0 for v in vals) and len(set(vals)) == 4:
                return block.astype(grid.dtype), (
                    np.zeros_like(grid, dtype=bool) |
                    (np.zeros((H, W)) == 0) &  # dummy to keep type
                    (
                        (np.arange(H)[:, None] >= r) &
                        (np.arange(H)[:, None] < r + 2) &
                        (np.arange(W)[None, :] >= c) &
                        (np.arange(W)[None, :] < c + 2)
                    )
                )
    return None, None


def _crop_mask(mask: np.ndarray, r0, r1, c0, c1):
    return mask[r0:r1 + 1, c0:c1 + 1]


def _transform_shape(mask: np.ndarray, mode: str):
    """
    Apply the appropriate geometric transformation to a component mask.
    """
    if mode == "identity":
        return mask
    if mode == "rot90_ccw":
        return np.rot90(mask, k=1)
    if mode == "rot180":
        return np.rot90(mask, k=2)
    if mode == "rot90_cw":
        return np.rot90(mask, k=-1)
    raise ValueError(f"Unknown transform mode: {mode}")


def _place_shape(out: np.ndarray, mask: np.ndarray, color: int, position: str):
    """
    Place mask into out anchored at one of the four corners.
    Only writes where output is still 0.
    """
    h, w = mask.shape
    H, W = out.shape

    if position == "tl":
        r0, c0 = 0, 0
    elif position == "tr":
        r0, c0 = 0, max(W - w, 0)
    elif position == "bl":
        r0, c0 = max(H - h, 0), 0
    elif position == "br":
        r0, c0 = max(H - h, 0), max(W - w, 0)
    else:
        raise ValueError(f"Unknown position: {position}")

    # If shape is larger than output, crop it to fit (safety for odd cases)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        crop_r0 = max(0, -r0)
        crop_c0 = max(0, -c0)
        crop_h = min(h, H - r0)
        crop_w = min(w, W - c0)

        if crop_h <= 0 or crop_w <= 0:
            return

        mask = mask[crop_r0:crop_r0 + crop_h, crop_c0:crop_c0 + crop_w]
        r0 = max(r0, 0)
        c0 = max(c0, 0)
        h, w = mask.shape

    target = out[r0:r0 + h, c0:c0 + w]
    mask = np.asarray(mask, dtype=bool)

    # Priority is enforced by only writing zeros.
    out[r0:r0 + h, c0:c0 + w] = np.where(target == 0 & mask, color, target)


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform ARC input to output.

    Rule:
      1. Find the 2x2 palette block.
      2. Find all colored components.
      3. The component containing the palette is the top-left palette shape.
      4. Assign remaining components to top-right, bottom-left, bottom-right
         palette colors by their relative spatial positions.
      5. Transform each shape:
             TL: identity
             TR: rotate 90 CCW
             BL: rotate 180
             BR: rotate 90 CW
      6. Place them into a corner-anchored output grid with priority:
             TL > TR > BR > BL
    """
    if grid.size == 0:
        return grid.copy()

    palette, palette_mask = _find_palette_2x2(grid)
    if palette is None:
        return grid.copy()

    components = _components(grid)
    if len(components) < 4:
        return np.zeros_like(grid)

    # Identify the palette-container component as the one whose component mask
    # contains the palette 2x2 block.
    container = None
    for comp in components:
        if np.all(comp["mask"][palette_mask]):
            container = comp
            break

    if container is None:
        # Fallback: component with largest area near palette center
        pr, pc = np.argwhere(palette_mask)
        pr0 = int(pr.min())
        pc0 = int(pc.min())
        best = None
        best_score = -1
        for comp in components:
            score = 0
            if pr0 >= comp["r0"] and pr0 <= comp["r1"] and pc0 >= comp["c0"] and pc0 <= comp["c1"]:
                score += 1000
            score -= abs(comp["r"] - pr0) + abs(comp["c"] - pc0)
            if score > best_score:
                best_score = score
                best = comp
        container = best

    if container is None:
        return np.zeros_like(grid)

    others = [c for c in components if c is not container]

    # Sort the other three components by their relative spatial quadrant.
    # Use centroid relative to the whole grid; robust enough for these tasks.
    H, W = grid.shape
    center_r, center_c = H / 2.0, W / 2.0

    def quadrant_key(comp):
        r, c = comp["r"], comp["c"]
        # Score for top-right, bottom-left, bottom-right
        tr = max(0.0, (c - center_c) / W + 1.0)
        bl = max(0.0, (r - center_r) / H + 1.0)
        br = max(0.0, (r - center_r) / H + (c - center_c) / W + 2.0)
        return tr, bl, br

    # We need three components: top-right, bottom-left, bottom-right.
    # Pick best by dominant direction.
    tr_comp = None
    bl_comp = None
    br_comp = None

    for comp in others:
        r, c = comp["r"], comp["c"]
        dr = r - center_r
        dc = c - center_c

        if dc >= 0 and dr < 0:
            if tr_comp is None or dc > tr_comp["c"] - center_c:
                tr_comp = comp
        elif dr >= 0 and dc <= 0:
            if bl_comp is None or dr > bl_comp["r"] - center_r:
                bl_comp = comp
        elif dr >= 0 and dc >= 0:
            if br_comp is None or (dr + dc) > ((br_comp["r"] - center_r) + (br_comp["c"] - center_c)):
                br_comp = comp

    # Fallbacks if quadrant detection is ambiguous:
    if tr_comp is None:
        # Topmost among remaining, biased right
        cands = [c for c in others if c is not bl_comp and c is not br_comp]
        if cands:
            tr_comp = max(cands, key=lambda x: -x["r"] + x["c"] / 10)
    if bl_comp is None:
        cands = [c for c in others if c is not tr_comp and c is not br_comp]
        if cands:
            bl_comp = max(cands, key=lambda x: x["r"] - x["c"] / 10)
    if br_comp is None:
        cands = [c for c in others if c is not tr_comp and c is not bl_comp]
        if cands:
            br_comp = max(cands, key=lambda x: x["r"] + x["c"])

    # If still missing, use remaining components by size as a last resort
    used = {id(container), id(tr_comp), id(bl_comp), id(br_comp)}
    remaining = [c for c in others if id(c) not in used]
    if tr_comp is None:
        tr_comp = remaining.pop(0) if remaining else None
    if bl_comp is None:
        bl_comp = remaining.pop(0) if remaining else None
    if br_comp is None:
        br_comp = remaining.pop(0) if remaining else None

    if any(x is None for x in (tr_comp, bl_comp, br_comp)):
        return np.zeros_like(grid)

    # Extract cropped masks and transform
    mask_tl = _crop_mask(container["mask"], container["r0"], container["r1"], container["c0"], container["c1"])
    mask_tr = _crop_mask(tr_comp["mask"], tr_comp["r0"], tr_comp["r1"], tr_comp["c0"], tr_comp["c1"])
    mask_bl = _crop_mask(bl_comp["mask"], bl_comp["r0"], bl_comp["r1"], bl_comp["c0"], bl_comp["c1"])
    mask_br = _crop_mask(br_comp["mask"], br_comp["r0"], br_comp["r1"], br_comp["c0"], br_comp["c1"])

    mask_tl = _transform_shape(mask_tl, "identity")
    mask_tr = _transform_shape(mask_tr, "rot90_ccw")
    mask_bl = _transform_shape(mask_bl, "rot180")
    mask_br = _transform_shape(mask_br, "rot90_cw")

    # Determine output canvas size:
    # The examples suggest the output is just large enough to host the four
    # corner-anchored shapes, plus one cell of separation if needed.
    # We use the bounding boxes of the four transformed shapes placed at corners.
    def corner_bbox(r0, c0, h, w, H, W):
        return (
            max(0, r0),
            max(0, c0),
            min(H, r0 + h),
            min(W, c0 + w),
        )

    # We need to find minimal H, W. Use an iterative but simple method:
    # start from max dimensions and grow until all shapes fit without negative anchors.
    # For these tasks, output is usually near max(shape_heights)+max(shape_widths).
    h_tl, w_tl = mask_tl.shape
    h_tr, w_tr = mask_tr.shape
    h_bl, w_bl = mask_bl.shape
    h_br, w_br = mask_br.shape

    # A good lower bound: each top/bottom pair must fit vertically, left/right pairs horizontally.
    H = max(h_tl + h_bl, h_tr + h_br, max(h_tl, h_tr, h_bl, h_br))
    W = max(w_tl + w_tr, w_bl + w_br, max(w_tl, w_tr, w_bl, w_br))

    # Ensure anchors are non-negative.
    H = max(H, h_tl, h_tr, h_bl, h_br)
    W = max(W, w_tl, w_tr, w_bl, w_br)

    out = np.zeros((H, W), dtype=grid.dtype)

    # Priority: TL first, then TR, BR, BL
    _place_shape(out, mask_tl, int(palette[0, 0]), "tl")
    _place_shape(out, mask_tr, int(palette[0, 1]), "tr")
    _place_shape(out, mask_br, int(palette[1, 1]), "br")
    _place_shape(out, mask_bl, int(palette[1, 0]), "bl")

    return out