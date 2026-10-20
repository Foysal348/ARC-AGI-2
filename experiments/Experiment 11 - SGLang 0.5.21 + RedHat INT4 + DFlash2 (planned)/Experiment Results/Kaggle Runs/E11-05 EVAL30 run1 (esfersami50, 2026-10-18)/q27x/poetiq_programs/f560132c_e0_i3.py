import numpy as np
from scipy import ndimage


def _bbox_crop(mask: np.ndarray) -> tuple:
    """Return (cropped_mask, r0, r1, c0, c1)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros_like(mask, dtype=bool), 0, -1, 0, -1
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    return mask[r0:r1 + 1, c0:c1 + 1], r0, r1, c0, c1


def _rotate90(mask: np.ndarray, direction: str) -> np.ndarray:
    """Rotate mask 90 degrees clockwise or counter-clockwise."""
    if direction == "cw":
        return np.rot90(mask, k=-1)
    if direction == "ccw":
        return np.rot90(mask, k=1)
    raise ValueError("direction must be 'cw' or 'ccw'")


def _place(out: np.ndarray, mask: np.ndarray, color: int, anchor: str) -> None:
    """Place mask into out at one of the four corners, writing only onto zeros."""
    h, w = mask.shape
    H, W = out.shape

    if anchor == "tl":
        r0, c0 = 0, 0
    elif anchor == "tr":
        r0, c0 = 0, max(W - w, 0)
    elif anchor == "bl":
        r0, c0 = max(H - h, 0), 0
    elif anchor == "br":
        r0, c0 = max(H - h, 0), max(W - w, 0)
    else:
        raise ValueError("anchor must be one of tl/tr/bl/br")

    if r0 >= H or c0 >= W:
        return

    r1 = min(H, r0 + h)
    c1 = min(W, c0 + w)
    mh = r1 - r0
    mw = c1 - c0

    if mh <= 0 or mw <= 0:
        return

    sub = out[r0:r1, c0:c1]
    msub = mask[:mh, :mw]
    out[r0:r1, c0:c1] = np.where((sub == 0) & msub, color, sub)


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Infer the four puzzle tiles from the input grid and compose them into
    a smaller output grid.

    Rule:
    1. Find the 2x2 palette block containing four distinct non-zero colors.
    2. The connected component containing that palette is the top-left tile.
    3. The other three components are assigned to the other corners:
         - top-right input -> bottom-left output
         - bottom-left input -> bottom-right output
         - bottom-right input -> top-right output
    4. Tile orientations:
         - top-left tile: identity
         - top-right tile: rotate 180
         - bottom-left tile: rotate 90 CW, then flip horizontally
         - bottom-right tile: rotate 90 CW
    5. Canvas size is determined so that the four tiles form a compact
       2x2 composition:
         H = top-row-height + bottom-row-height
         W = left-column-width + right-column-width
       where the top/bottom and left/right contributions are chosen from the
       transformed tile dimensions.
    6. Tiles are placed with priority:
         top-left > top-right > bottom-right > bottom-left.
    """
    if grid.size == 0:
        return grid.copy()

    H_in, W_in = grid.shape

    # ---------------------------------------------------------------------
    # 1. Find the palette block: a 2x2 block of four distinct non-zero colors.
    # ---------------------------------------------------------------------
    palette = None
    palette_pos = None

    for r in range(H_in - 1):
        for c in range(W_in - 1):
            block = grid[r:r + 2, c:c + 2]
            vals = block.flatten().tolist()
            if all(v != 0 for v in vals) and len(set(vals)) == 4:
                palette = block.astype(grid.dtype)
                palette_pos = (r, c)
                break
        if palette is not None:
            break

    if palette is None:
        return np.zeros_like(grid)

    color_tl = int(palette[0, 0])
    color_tr = int(palette[0, 1])
    color_bl = int(palette[1, 0])
    color_br = int(palette[1, 1])

    # ---------------------------------------------------------------------
    # 2. Extract connected components of all non-zero pixels.
    # ---------------------------------------------------------------------
    labeled, n = ndimage.label(grid != 0)
    comps = []

    for idx in range(1, n + 1):
        mask = labeled == idx
        mask_crop, r0, r1, c0, c1 = _bbox_crop(mask)
        if r1 < r0:
            continue
        comps.append({
            "mask": mask,
            "crop": mask_crop,
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": c1,
            "r": (r0 + r1) / 2.0,
            "c": (c0 + c1) / 2.0,
        })

    if len(comps) < 4:
        return np.zeros_like(grid)

    # ---------------------------------------------------------------------
    # 3. Identify the component containing the palette.
    # ---------------------------------------------------------------------
    pr, pc = palette_pos

    container = None
    for comp in comps:
        if (
            comp["r0"] <= pr <= comp["r1"]
            and comp["c0"] <= pc <= comp["c1"]
        ):
            container = comp
            break

    if container is None:
        # Fallback: component closest to the palette.
        best = None
        best_d = float("inf")
        for comp in comps:
            d = abs(comp["r"] - pr) + abs(comp["c"] - pc)
            if d < best_d:
                best_d = d
                best = comp
        container = best

    others = [c for c in comps if c is not container]

    # ---------------------------------------------------------------------
    # 4. Assign the other three components to input quadrants
    #    relative to the whole grid.
    # ---------------------------------------------------------------------
    mid_r = H_in / 2.0
    mid_c = W_in / 2.0

    tr_comp = None
    bl_comp = None
    br_comp = None

    for comp in others:
        dr = comp["r"] - mid_r
        dc = comp["c"] - mid_c

        if dr < 0 and dc >= 0:
            tr_comp = comp
        elif dr >= 0 and dc < 0:
            bl_comp = comp
        elif dr >= 0 and dc >= 0:
            br_comp = comp
        elif dr < 0 and dc < 0:
            # Ambiguous: treat the topmost/rightmost among remaining as TR
            # if needed, otherwise ignore.
            if tr_comp is None:
                tr_comp = comp
        else:
            # Should not happen.
            pass

    # Fallbacks if quadrant detection failed.
    used_ids = {id(container)}
    if tr_comp is not None:
        used_ids.add(id(tr_comp))
    if bl_comp is not None:
        used_ids.add(id(bl_comp))
    if br_comp is not None:
        used_ids.add(id(br_comp))

    remaining = [c for c in others if id(c) not in used_ids]
    if tr_comp is None:
        tr_comp = remaining.pop(0) if remaining else None
    if bl_comp is None:
        bl_comp = remaining.pop(0) if remaining else None
    if br_comp is None:
        br_comp = remaining.pop(0) if remaining else None

    if any(x is None for x in (tr_comp, bl_comp, br_comp)):
        return np.zeros_like(grid)

    # ---------------------------------------------------------------------
    # 5. Prepare transformed masks for the four output tiles.
    # ---------------------------------------------------------------------
    mask_tl = container["crop"]
    mask_tr = np.rot90(tr_comp["crop"], k=2)          # rotate 180
    mask_bl = np.flip(_rotate90(bl_comp["crop"], "cw"), axis=1)  # 90 CW + horizontal flip
    mask_br = _rotate90(br_comp["crop"], "cw")

    h_tl, w_tl = mask_tl.shape
    h_tr, w_tr = mask_tr.shape
    h_bl, w_bl = mask_bl.shape
    h_br, w_br = mask_br.shape

    # ---------------------------------------------------------------------
    # 6. Determine output canvas size.
    #
    # In the examples, the top row height is the larger of the top-left
    # and top-right tile heights, and the bottom row height is the larger of
    # bottom-left and bottom-right. Similarly for widths.
    # ---------------------------------------------------------------------
    top_h = max(h_tl, h_tr)
    bot_h = max(h_bl, h_br)
    left_w = max(w_tl, w_bl)
    right_w = max(w_tr, w_br)

    H_out = top_h + bot_h
    W_out = left_w + right_w

    # Safety: ensure at least enough size for each tile.
    H_out = max(H_out, h_tl, h_tr, h_bl, h_br)
    W_out = max(W_out, w_tl, w_tr, w_bl, w_br)

    out = np.zeros((H_out, W_out), dtype=grid.dtype)

    # ---------------------------------------------------------------------
    # 7. Place tiles with priority:
    #    TL > TR > BR > BL
    # ---------------------------------------------------------------------
    _place(out, mask_tl, color_tl, "tl")
    _place(out, mask_tr, color_tr, "tr")
    _place(out, mask_br, color_br, "br")
    _place(out, mask_bl, color_bl, "bl")

    return out