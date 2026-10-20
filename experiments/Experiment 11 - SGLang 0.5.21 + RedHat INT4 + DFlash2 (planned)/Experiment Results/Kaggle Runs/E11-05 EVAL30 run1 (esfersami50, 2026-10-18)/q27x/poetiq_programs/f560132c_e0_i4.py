import numpy as np
from scipy import ndimage


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform an ARC grid by extracting the four non-background objects,
    re-coloring/rotating them according to a 2x2 palette, and packing them
    into a compact rectangle.

    Rule inferred from the examples:
    1. Find a 2x2 palette block containing four distinct non-zero colors.
       The palette is:
           [[TL, TR],
            [BL, BR]]
    2. The connected component containing the palette is the top-left tile.
       It is used unchanged and colored with palette[TL].
    3. The other three components are assigned to the remaining output
       quadrants by input position:
           - top-right input    -> bottom-left output, rotated 180
           - bottom-left input  -> bottom-right output, rotated 90 CW
           - bottom-right input -> top-right output, rotated 90 CCW
    4. The final canvas is the smallest square-ish rectangle large enough to
       hold the transformed tiles when anchored to the four corners.
       In the ARC examples this is the minimal size where the four colored
       regions tile the output without leaving background cells.
    """
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ---------------------------------------------------------------------
    # 1. Find the 2x2 palette.
    # ---------------------------------------------------------------------
    palette = None
    palette_pos = None
    for r in range(H - 1):
        for c in range(W - 1):
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
    # 2. Connected components of non-background pixels.
    # ---------------------------------------------------------------------
    labeled, n = ndimage.label(grid != 0)
    comps = []
    for idx in range(1, n + 1):
        mask = labeled == idx
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        r0, r1 = int(ys.min()), int(ys.max())
        c0, c1 = int(xs.min()), int(xs.max())
        crop = mask[r0:r1 + 1, c0:c1 + 1]
        comps.append({
            "crop": crop,
            "r0": r0, "r1": r1, "c0": c0, "c1": c1,
            "r": (r0 + r1) / 2.0,
            "c": (c0 + c1) / 2.0,
        })

    if len(comps) < 4:
        return np.zeros_like(grid)

    # ---------------------------------------------------------------------
    # 3. Identify the component containing the palette.
    # ---------------------------------------------------------------------
    pr, pc = palette_pos
    tl_comp = None
    for comp in comps:
        if comp["r0"] <= pr <= comp["r1"] and comp["c0"] <= pc <= comp["c1"]:
            tl_comp = comp
            break

    if tl_comp is None:
        # Fallback: nearest component.
        tl_comp = min(comps, key=lambda x: abs(x["r"] - pr) + abs(x["c"] - pc))

    others = [c for c in comps if c is not tl_comp]

    # ---------------------------------------------------------------------
    # 4. Assign remaining components to input quadrants relative to grid center.
    # ---------------------------------------------------------------------
    mid_r, mid_c = H / 2.0, W / 2.0
    tr_comp = bl_comp = br_comp = None

    for comp in others:
        dr = comp["r"] - mid_r
        dc = comp["c"] - mid_c

        if dr < 0 and dc >= 0:
            if tr_comp is None or dc > tr_comp["c"] - mid_c:
                tr_comp = comp
        elif dr >= 0 and dc < 0:
            if bl_comp is None or dr > bl_comp["r"] - mid_r:
                bl_comp = comp
        elif dr >= 0 and dc >= 0:
            if br_comp is None or (dr + dc) > ((br_comp["r"] - mid_r) + (br_comp["c"] - mid_c)):
                br_comp = comp
        else:
            # Fallback if an object lies near center or in ambiguous quadrant.
            if tr_comp is None:
                tr_comp = comp

    # Fill any missing slots with unused components, preferring by position.
    used = {id(tl_comp), id(tr_comp), id(bl_comp), id(br_comp)}
    unused = [c for c in others if id(c) not in used]

    if tr_comp is None:
        tr_comp = unused.pop(0) if unused else None
    if bl_comp is None:
        bl_comp = unused.pop(0) if unused else None
    if br_comp is None:
        br_comp = unused.pop(0) if unused else None

    if any(x is None for x in (tr_comp, bl_comp, br_comp)):
        return np.zeros_like(grid)

    # ---------------------------------------------------------------------
    # 5. Apply transforms.
    # ---------------------------------------------------------------------
    mask_tl = tl_comp["crop"]
    mask_tr = np.rot90(br_comp["crop"], k=1)      # 90 CCW
    mask_bl = np.rot90(tr_comp["crop"], k=2)      # 180
    mask_br = np.rot90(bl_comp["crop"], k=-1)     # 90 CW

    h_tl, w_tl = mask_tl.shape
    h_tr, w_tr = mask_tr.shape
    h_bl, w_bl = mask_bl.shape
    h_br, w_br = mask_br.shape

    # ---------------------------------------------------------------------
    # 6. Determine a compact output size.
    #
    # The examples use the minimal square/rectangle that allows the four
    # corner-anchored masks to fill the output without holes.
    # Try small square sizes and choose the first one where all four masks
    # can be placed in their respective corners without overlap.
    # ---------------------------------------------------------------------
    def fits_at_corner(mask, color_anchor):
        h, w = mask.shape
        H_out, W_out = h + w, h + w  # temporary not used, just shape
        return True

    def can_place(H_out, W_out):
        # TL at (0,0)
        if h_tl > H_out or w_tl > W_out:
            return False
        # TR at (0, W_out - w_tr)
        if h_tr > H_out or w_tr > W_out:
            return False
        # BL at (H_out - h_bl, 0)
        if h_bl > H_out or w_bl > W_out:
            return False
        # BR at (H_out - h_br, W_out - w_br)
        if h_br > H_out or w_br > W_out:
            return False

        # Build occupancy and check overlaps.
        occ = np.zeros((H_out, W_out), dtype=bool)

        def place(mask, r0, c0):
            h, w = mask.shape
            r1 = min(H_out, r0 + h)
            c1 = min(W_out, c0 + w)
            if r0 < 0 or c0 < 0 or r1 <= r0 or c1 <= c0:
                return False
            sub = occ[r0:r1, c0:c1]
            m = mask[:r1 - r0, :c1 - c0]
            if np.any(sub & m):
                return False
            occ[r0:r1, c0:c1] |= m
            return True

        if not place(mask_tl, 0, 0):
            return False
        if not place(mask_tr, 0, W_out - w_tr):
            return False
        if not place(mask_br, H_out - h_br, W_out - w_br):
            return False
        if not place(mask_bl, H_out - h_bl, 0):
            return False
        return True

    # Candidate output sizes: start from minimal dimensions, then expand.
    min_h = max(h_tl + h_bl, h_tr + h_br, h_tl, h_tr, h_bl, h_br)
    min_w = max(w_tl + w_tr, w_bl + w_br, w_tl, w_tr, w_bl, w_br)

    best = None
    # Search a modest range around the natural size.
    for H_out in range(max(1, min_h), max(min_h + 15, 40)):
        for W_out in range(max(1, min_w), max(min_w + 15, 40)):
            if can_place(H_out, W_out):
                # Prefer the smallest area; among equals prefer smaller max dimension.
                best = (H_out, W_out)
                break
        if best is not None:
            break

    if best is None:
        # Fallback: simply use maximal row/column sums.
        H_out = max(h_tl + h_bl, h_tr + h_br, h_tl, h_tr, h_bl, h_br)
        W_out = max(w_tl + w_tr, w_bl + w_br, w_tl, w_tr, w_bl, w_br)
    else:
        H_out, W_out = best

    out = np.zeros((H_out, W_out), dtype=grid.dtype)

    # ---------------------------------------------------------------------
    # 7. Place tiles. Order is important only if unexpected overlaps occur.
    # ---------------------------------------------------------------------
    def put(mask, color, r0, c0):
        h, w = mask.shape
        r1 = min(H_out, r0 + h)
        c1 = min(W_out, c0 + w)
        if r0 < 0 or c0 < 0:
            return
        out[r0:r1, c0:c1] = np.where(mask[:r1 - r0, :c1 - c0], color, out[r0:r1, c0:c1])

    put(mask_tl, color_tl, 0, 0)
    put(mask_tr, color_tr, 0, W_out - w_tr)
    put(mask_bl, color_bl, H_out - h_bl, 0)
    put(mask_br, color_br, H_out - h_br, W_out - w_br)

    # In the examples, the fourth color also fills any remaining background
    # cells to produce a completely tiled rectangle.
    # This matches both provided examples exactly.
    filler_color = color_br if np.count_nonzero(out == 0) > 0 else 0
    if filler_color:
        out[out == 0] = filler_color

    return out