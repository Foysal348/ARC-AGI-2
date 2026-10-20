import numpy as np
import math

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the three non-source quadrants from the top-left source pattern.

    Rule:
    - The top-left quadrant contains the original source pattern.
    - The vertical and horizontal 1-lines are fold/mirror axes.
    - The other quadrants contain small 2-blocks that specify the size/position
      of the corresponding 2-block after mirroring and scaling.
    - For each quadrant:
        * top-right: horizontal flip of source
        * bottom-left: vertical flip of source
        * bottom-right: 180-degree rotation of source
      Then resize the transformed source so that its 2-block matches the hint
      2-block size, and align the transformed 2-block top-left to the hint.
    """
    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, copy=True)
    H, W = grid.shape
    mid_r = H // 2
    mid_c = W // 2

    # Background is color 4.
    BG = 4
    HINT = 2

    # Extract the source pattern from the top-left quadrant.
    tl = grid[:mid_r, :mid_c]
    non_bg = np.argwhere(tl != BG)

    if non_bg.size == 0:
        return grid

    r0, c0 = non_bg[:, 0].min(), non_bg[:, 1].min()
    r1, c1 = non_bg[:, 0].max(), non_bg[:, 1].max()
    source = tl[r0:r1 + 1, c0:c1 + 1].copy()

    # Find the 2-block in the source.
    twos = np.argwhere(source == HINT)
    if twos.size == 0:
        # If no 2-block exists, simply mirror without scaling.
        sr2 = sc2 = sh2 = sw2 = 0
    else:
        sr2 = twos[:, 0].min()
        sr2_max = twos[:, 0].max()
        sc2 = twos[:, 1].min()
        sc2_max = twos[:, 1].max()
        sh2 = sr2_max - sr2 + 1
        sw2 = sc2_max - sc2 + 1

    sh, sw = source.shape

    def nearest_resize(arr: np.ndarray, scale: float) -> np.ndarray:
        """
        Nearest-neighbor resize by a (possibly fractional) scale.
        Output pixel i samples source pixel floor(i / scale).
        """
        if scale <= 0:
            scale = 1.0
        nh = max(1, int(round(arr.shape[0] * scale)))
        nw = max(1, int(round(arr.shape[1] * scale)))
        out = np.empty((nh, nw), dtype=arr.dtype)

        for i in range(nh):
            src_r = min(arr.shape[0] - 1, int(math.floor(i / scale + 1e-12)))
            for j in range(nw):
                src_c = min(arr.shape[1] - 1, int(math.floor(j / scale + 1e-12)))
                out[i, j] = arr[src_r, src_c]
        return out

    def scaled_block_top(patt: np.ndarray, scale: float, r2: int, c2: int, h2: int, w2: int):
        """
        Return the top-left coordinates of the scaled image of the source 2-block.
        """
        ph, pw = patt.shape
        best_r = None
        best_c = None

        for i in range(ph):
            src_r = min(sh - 1, int(math.floor(i / scale + 1e-12)))
            if r2 <= src_r < r2 + h2:
                best_r = i
                break
        for j in range(pw):
            src_c = min(sw - 1, int(math.floor(j / scale + 1e-12)))
            if c2 <= src_c < c2 + w2:
                best_c = j
                break

        if best_r is None:
            best_r = min(ph - 1, int(math.ceil(r2 * scale)))
        if best_c is None:
            best_c = min(pw - 1, int(math.ceil(c2 * scale)))

        return best_r, best_c

    def find_hint(quad: np.ndarray):
        """Find bounding box of 2-hint in a quadrant, or None."""
        pts = np.argwhere(quad == HINT)
        if pts.size == 0:
            return None
        rmin = pts[:, 0].min()
        rmax = pts[:, 0].max()
        cmin = pts[:, 1].min()
        cmax = pts[:, 1].max()
        return rmin, cmin, rmax - rmin + 1, cmax - cmin + 1

    def place_pattern(out: np.ndarray, quad_r0, quad_r1, quad_c0, quad_c1,
                      patt: np.ndarray, scale: float,
                      sr2, sc2, sh2, sw2,
                      hint_global_r, hint_global_c):
        """Place scaled pattern so its 2-block aligns to hint top-left."""
        if sh2 == 0 or sw2 == 0:
            return

        p2_r, p2_c = scaled_block_top(patt, scale, sr2, sc2, sh2, sw2)

        offset_r = hint_global_r - p2_r
        offset_c = hint_global_c - p2_c

        # Clear this quadrant first.
        out[quad_r0:quad_r1, quad_c0:quad_c1] = BG

        ph, pw = patt.shape
        for i in range(ph):
            gr = offset_r + i
            if gr < quad_r0 or gr >= quad_r1:
                continue
            for j in range(pw):
                gc = offset_c + j
                if gc < quad_c0 or gc >= quad_c1:
                    continue
                out[gr, gc] = patt[i, j]

    # Top-right: horizontal flip
    tr_quad = grid[:mid_r, mid_c + 1:]
    hint_tr = find_hint(tr_quad)
    if hint_tr is not None:
        hr, hc, hh, hw = hint_tr
        if sh2 and sw2:
            s = max(hh / sh2, hw / sw2)
        else:
            s = 1.0
        patt = np.fliplr(source)
        place_pattern(
            grid,
            0, mid_r, mid_c + 1, W,
            patt, s,
            sr2, sw - (sc2 + sw2) if sw2 else 0,
            sh2, sw2,
            hr, mid_c + 1 + hc
        )

    # Bottom-left: vertical flip
    bl_quad = grid[mid_r + 1:, :mid_c]
    hint_bl = find_hint(bl_quad)
    if hint_bl is not None:
        hr, hc, hh, hw = hint_bl
        if sh2 and sw2:
            s = max(hh / sh2, hw / sw2)
        else:
            s = 1.0
        patt = np.flipud(source)
        place_pattern(
            grid,
            mid_r + 1, H, 0, mid_c,
            patt, s,
            sh - (sr2 + sh2) if sh2 else 0, sc2,
            sh2, sw2,
            mid_r + 1 + hr, hc
        )

    # Bottom-right: 180 rotation
    br_quad = grid[mid_r + 1:, mid_c + 1:]
    hint_br = find_hint(br_quad)
    if hint_br is not None:
        hr, hc, hh, hw = hint_br
        if sh2 and sw2:
            s = max(hh / sh2, hw / sw2)
        else:
            s = 1.0
        patt = np.flipud(np.fliplr(source))
        place_pattern(
            grid,
            mid_r + 1, H, mid_c + 1, W,
            patt, s,
            sh - (sr2 + sh2) if sh2 else 0,
            sw - (sc2 + sw2) if sw2 else 0,
            sh2, sw2,
            mid_r + 1 + hr, mid_c + 1 + hc
        )

    return grid