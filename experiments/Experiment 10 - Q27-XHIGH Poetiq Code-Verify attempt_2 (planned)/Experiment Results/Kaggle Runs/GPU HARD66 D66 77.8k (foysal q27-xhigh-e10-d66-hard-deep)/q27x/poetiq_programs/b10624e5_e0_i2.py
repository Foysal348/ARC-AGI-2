import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete the three target quadrants from the top-left source pattern.

    Rule
    ----
    The grid is split into four quadrants by the central row/column of 1s.
    The top-left quadrant contains the source pattern. The other three
    quadrants contain a color-2 rectangle that acts as a hint.

    For each of the three target quadrants:
      * top-right: horizontally flip the source pattern
      * bottom-left: vertically flip the source pattern
      * bottom-right: rotate the source pattern by 180 degrees

    The color-2 rectangle in the source pattern is the reference block.
    The hint color-2 rectangle in a target quadrant determines the size and
    position of that reference block after transformation.  The whole source
    pattern is therefore resized and translated so that its transformed
    color-2 block exactly coincides with the hint rectangle.

    The implementation uses a block-aware nearest-neighbor resize:
      * the reference 2-block is stretched exactly to the hint size
      * other source pixels are sampled with a nearest-neighbor mapping
      * the transformed pattern is placed so that the transformed 2-block
        matches the hint position exactly
    """

    if grid.size == 0:
        return grid.copy()

    grid = np.array(grid, copy=True)
    H, W = grid.shape
    mid_r, mid_c = H // 2, W // 2

    # Background color is the dominant non-separator color in the top-left
    # quadrant. In the given tasks this is color 4, but we detect it robustly.
    tl = grid[:mid_r, :mid_c]
    colors, counts = np.unique(tl, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    HINT_COLOR = 2

    # ------------------------------------------------------------------
    # Extract source pattern from the top-left quadrant.
    # ------------------------------------------------------------------
    non_bg = np.argwhere(tl != bg)
    if non_bg.size == 0:
        return grid

    r0, c0 = non_bg[:, 0].min(), non_bg[:, 1].min()
    r1, c1 = non_bg[:, 0].max(), non_bg[:, 1].max()
    source = tl[r0:r1 + 1, c0:c1 + 1].copy()

    sh_src, sw_src = source.shape

    # Reference 2-block in the source pattern.
    ref_pts = np.argwhere(source == HINT_COLOR)
    if ref_pts.size == 0:
        return grid

    sr_ref, sc_ref = ref_pts[:, 0].min(), ref_pts[:, 1].min()
    sr_ref1, sc_ref1 = ref_pts[:, 0].max(), ref_pts[:, 1].max()
    sr_ref_h = sr_ref1 - sr_ref + 1
    sr_ref_w = sc_ref1 - sc_ref + 1

    # ------------------------------------------------------------------
    # Helper functions.
    # ------------------------------------------------------------------
    def find_2_box(quad: np.ndarray):
        """Bounding box of all 2s in a quadrant, or None."""
        pts = np.argwhere(quad == HINT_COLOR)
        if pts.size == 0:
            return None
        rmin, cmin = pts[:, 0].min(), pts[:, 1].min()
        rmax, cmax = pts[:, 0].max(), pts[:, 1].max()
        return int(rmin), int(cmin), int(rmax - rmin + 1), int(cmax - cmin + 1)

    def ref_box_after_transform(patt: np.ndarray):
        """
        Return top-left and size of the reference 2-block in the transformed
        pattern `patt`.  The pattern is assumed to be one of:
          fliplr(source), flipud(source), flipud(fliplr(source)).
        """
        # Easiest: directly find 2s in the transformed pattern.
        pts = np.argwhere(patt == HINT_COLOR)
        if pts.size == 0:
            return 0, 0, 0, 0
        rmin, cmin = pts[:, 0].min(), pts[:, 1].min()
        rmax, cmax = pts[:, 0].max(), pts[:, 1].max()
        return int(rmin), int(cmin), int(rmax - rmin + 1), int(cmax - cmin + 1)

    def resized_box(patt: np.ndarray, rr, rc, rh, rw, out_h, out_w):
        """
        Compute the bounding box in the resized output of a rectangular box
        from the input pattern.

        Uses the same nearest-neighbor mapping as `resize_pattern`:
            source_coord = floor(output_coord * src_size / out_size)
        """
        ph, pw = patt.shape

        # Rows.
        def map_range(src_start, src_end, src_size, out_size):
            # First output index that maps into [src_start, src_end]
            a = None
            for i in range(out_size):
                s = min(src_size - 1, int(np.floor(i * src_size / out_size + 1e-12)))
                if src_start <= s < src_end:
                    a = i
                    break
            if a is None:
                # Fallback for tiny/rounding edge cases.
                a = max(0, min(out_size - 1, int(np.floor(src_start * out_size / src_size))))

            b = None
            for i in range(out_size - 1, -1, -1):
                s = min(src_size - 1, int(np.floor(i * src_size / out_size + 1e-12)))
                if src_start <= s < src_end:
                    b = i
                    break
            if b is None:
                b = max(a, min(out_size - 1, int(np.floor((src_end - 1) * out_size / src_size))))

            return a, b

        r_a, r_b = map_range(rr, rr + rh, ph, out_h)
        c_a, c_b = map_range(rc, rc + rw, pw, out_w)

        return r_a, c_a, r_b - r_a + 1, c_b - c_a + 1

    def resize_pattern(patt: np.ndarray, target_ref_h: int, target_ref_w: int):
        """
        Resize the whole pattern so that its 2-block becomes target_ref_h x
        target_ref_w.  The output size is chosen as the smallest size that can
        represent the reference block at the requested size while still giving
        every source row/column at least one output pixel.
        """
        ph, pw = patt.shape
        rr, rc, rh, rw = ref_box_after_transform(patt)

        if rh == 0 or rw == 0:
            return patt.copy(), (0, 0, 0, 0)

        # Choose output dimensions.
        # Minimum size must be at least the target reference size.
        # It must also be large enough that, under nearest-neighbor sampling,
        # the reference box maps to at least the target height/width.
        # We also ensure every source row/column gets at least one sample.
        def choose_out_size(src_size, ref_start, ref_len, target_len):
            # Start with the natural continuous-scale size.
            scale_h = target_len / ref_len
            natural = max(1, int(round(src_size * scale_h)))
            # Smallest size that lets the reference interval reach target_len
            # while keeping at least one pixel per source coordinate.
            out_size = max(natural, target_len, src_size)

            # Increase until the nearest-neighbor mapping gives at least
            # target_len pixels inside the reference box.
            while out_size < max(src_size * 4, 16):
                a = None
                b = None
                for i in range(out_size):
                    s = min(src_size - 1, int(np.floor(i * src_size / out_size + 1e-12)))
                    if ref_start <= s < ref_start + ref_len:
                        if a is None:
                            a = i
                        b = i
                if a is not None and (b - a + 1) >= target_len:
                    break
                out_size += 1

            return out_size

        out_h = choose_out_size(ph, rr, rh, target_ref_h)
        out_w = choose_out_size(pw, rc, rw, target_ref_w)

        out = np.empty((out_h, out_w), dtype=patt.dtype)
        for i in range(out_h):
            sr = min(ph - 1, int(np.floor(i * ph / out_h + 1e-12)))
            for j in range(out_w):
                sc = min(pw - 1, int(np.floor(j * pw / out_w + 1e-12)))
                out[i, j] = patt[sr, sc]

        # Compute where the transformed 2-block landed in the resized pattern.
        box = resized_box(patt, rr, rc, rh, rw, out_h, out_w)
        return out, box

    def place_pattern(out_grid: np.ndarray,
                      quad_r0, quad_r1, quad_c0, quad_c1,
                      pattern: np.ndarray,
                      pattern_2_box,
                      hint_global_r, hint_global_c):
        """
        Place `pattern` into the target quadrant so that its 2-block matches
        the hint location.
        """
        pr, pc, _, _ = pattern_2_box
        offset_r = hint_global_r - pr
        offset_c = hint_global_c - pc

        # Clear only the target quadrant, leaving the central 1-cross intact.
        out_grid[quad_r0:quad_r1, quad_c0:quad_c1] = bg

        ph, pw = pattern.shape
        for i in range(ph):
            gr = offset_r + i
            if gr < quad_r0 or gr >= quad_r1:
                continue
            for j in range(pw):
                gc = offset_c + j
                if gc < quad_c0 or gc >= quad_c1:
                    continue
                out_grid[gr, gc] = pattern[i, j]

    # ------------------------------------------------------------------
    # Process each target quadrant.
    # ------------------------------------------------------------------

    # Top-right: horizontal flip.
    tr_quad = grid[:mid_r, mid_c + 1:]
    hint_tr = find_2_box(tr_quad)
    if hint_tr is not None:
        hr, hc, hh, hw = hint_tr
        hint_gr, hint_gc = hr, mid_c + 1 + hc
        patt = np.fliplr(source)
        resized, ref_box = resize_pattern(patt, hh, hw)
        place_pattern(
            grid,
            0, mid_r,
            mid_c + 1, W,
            resized,
            ref_box,
            hint_gr,
            hint_gc
        )

    # Bottom-left: vertical flip.
    bl_quad = grid[mid_r + 1:, :mid_c]
    hint_bl = find_2_box(bl_quad)
    if hint_bl is not None:
        hr, hc, hh, hw = hint_bl
        hint_gr, hint_gc = mid_r + 1 + hr, hc
        patt = np.flipud(source)
        resized, ref_box = resize_pattern(patt, hh, hw)
        place_pattern(
            grid,
            mid_r + 1, H,
            0, mid_c,
            resized,
            ref_box,
            hint_gr,
            hint_gc
        )

    # Bottom-right: 180-degree rotation.
    br_quad = grid[mid_r + 1:, mid_c + 1:]
    hint_br = find_2_box(br_quad)
    if hint_br is not None:
        hr, hc, hh, hw = hint_br
        hint_gr, hint_gc = mid_r + 1 + hr, mid_c + 1 + hc
        patt = np.flipud(np.fliplr(source))
        resized, ref_box = resize_pattern(patt, hh, hw)
        place_pattern(
            grid,
            mid_r + 1, H,
            mid_c + 1, W,
            resized,
            ref_box,
            hint_gr,
            hint_gc
        )

    return grid