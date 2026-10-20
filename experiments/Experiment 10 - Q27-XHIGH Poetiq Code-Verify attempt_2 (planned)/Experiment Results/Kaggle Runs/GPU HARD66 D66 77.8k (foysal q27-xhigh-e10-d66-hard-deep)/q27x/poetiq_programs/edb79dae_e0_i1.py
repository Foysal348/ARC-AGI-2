import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the challenge grid by:
    1. Cropping to the rectangular frame made of 5s.
    2. Identifying the background color inside that frame.
    3. Finding the solid colored blocks inside the frame.
    4. Reading color remapping rules from 3x4 indicators of the form:
         C C D D
         C C C D
         C C D D
       which means color C should be recolored to D.
    5. For each block color, find the corresponding pure colored template
       outside the frame and use its pattern to fill the block, recolored
       according to the rule.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    # ------------------------------------------------------------------
    # Locate the main frame.  In all provided cases it is color 5.
    # ------------------------------------------------------------------
    frame_color = 5
    if not np.any(grid == frame_color):
        vals, cnts = np.unique(grid, return_counts=True)
        frame_color = vals[np.argmax(cnts)]

    ys, xs = np.argwhere(grid == frame_color).T
    if len(ys) == 0:
        return grid.copy()

    fr0, fr1 = int(ys.min()), int(ys.max())
    fc0, fc1 = int(xs.min()), int(xs.max())

    out = grid[fr0:fr1 + 1, fc0:fc1 + 1].copy()

    # ------------------------------------------------------------------
    # Background color is the most frequent non-frame color inside frame.
    # ------------------------------------------------------------------
    interior = out[1:-1, 1:-1]
    if interior.size == 0:
        bg = 0
    else:
        vals, cnts = np.unique(interior[interior != frame_color], return_counts=True)
        if len(vals) == 0:
            bg = 0
        else:
            bg = vals[np.argmax(cnts)]

    # ------------------------------------------------------------------
    # Find solid rectangular blocks inside the frame.
    # ------------------------------------------------------------------
    H, W = out.shape
    mask = (out != bg) & (out != frame_color)
    visited = np.zeros((H, W), dtype=bool)
    blocks = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or visited[r, c]:
                continue

            color = int(out[r, c])
            if color == bg or color == frame_color:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            min_r = max_r = r
            min_c = max_c = c
            count = 0

            while stack:
                cr, cc = stack.pop()
                count += 1
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if not visited[nr, nc] and out[nr, nc] == color:
                            visited[nr, nc] = True
                            stack.append((nr, nc))

            bh = max_r - min_r + 1
            bw = max_c - min_c + 1

            # Keep only solid rectangular components.
            if count == bh * bw:
                blocks.append((min_r, min_c, bh, bw, color))

    # ------------------------------------------------------------------
    # Find color remapping rules from 3x4 two-color indicators.
    # ------------------------------------------------------------------
    pattern = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=bool,
    )

    remap_votes = {}

    def vote(src, dst):
        remap_votes.setdefault(src, {})
        remap_votes[src][dst] = remap_votes[src].get(dst, 0) + 1

    GH, GW = grid.shape
    for r in range(GH - 2):
        for c in range(GW - 3):
            win = grid[r:r + 3, c:c + 4]
            vals = np.unique(win)

            if len(vals) != 2:
                continue
            if frame_color in vals:
                continue
            # Indicators use two non-background colors.
            if bg in vals:
                continue

            a, b = int(vals[0]), int(vals[1])
            if np.array_equal(win == a, pattern):
                vote(a, b)
            if np.array_equal(win == b, pattern):
                vote(b, a)

    remap = {}
    for src, dsts in remap_votes.items():
        if dsts:
            # Most frequent destination wins; lower color as tie-breaker.
            remap[src] = max(dsts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    # ------------------------------------------------------------------
    # Template lookup.
    #
    # A valid template for color x is a connected component of x whose
    # bounding box contains only x and the background color, and contains
    # at least one hole.  If the component is larger than the block,
    # use its top-left block-sized slice.
    # ------------------------------------------------------------------
    comp_cache = {}

    def get_components(color):
        if color in comp_cache:
            return comp_cache[color]

        target = grid == color
        vis = np.zeros((GH, GW), dtype=bool)
        comps = []

        for r in range(GH):
            for c in range(GW):
                if not target[r, c] or vis[r, c]:
                    continue

                stack = [(r, c)]
                vis[r, c] = True
                min_r = max_r = r
                min_c = max_c = c
                count = 0

                while stack:
                    cr, cc = stack.pop()
                    count += 1
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < GH and 0 <= nc < GW:
                            if target[nr, nc] and not vis[nr, nc]:
                                vis[nr, nc] = True
                                stack.append((nr, nc))

                ch = max_r - min_r + 1
                cw = max_c - min_c + 1
                area = ch * cw

                sub = grid[min_r:max_r + 1, min_c:max_c + 1]
                pure = np.all((sub == color) | (sub == bg))
                has_hole = count < area

                inside_frame = (
                    min_r >= fr0 and max_r <= fr1 and
                    min_c >= fc0 and max_c <= fc1
                )

                if pure and has_hole:
                    comps.append((area, min_r, min_c, ch, cw, inside_frame))

        comp_cache[color] = comps
        return comps

    def get_template_mask(color, h, w):
        comps = get_components(color)

        candidates = []
        for area, r0, c0, ch, cw, inside in comps:
            if ch >= h and cw >= w and not inside:
                candidates.append((area, r0, c0))

        # Fallback: if none outside, allow inside pattern components.
        if not candidates:
            for area, r0, c0, ch, cw, inside in comps:
                if ch >= h and cw >= w:
                    candidates.append((area, r0, c0))

        if not candidates:
            return None

        # Prefer the smallest sufficiently large template, then top-left.
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        _, r0, c0 = candidates[0]

        mask = (grid[r0:r0 + h, c0:c0 + w] == color)

        # If the top-left slice is degenerate, search within the component
        # for a slice that actually contains both color and background.
        if mask.all() or not mask.any():
            ch = r0 + h
            cw = c0 + w
            for dr in range(h + 1):
                for dc in range(w + 1):
                    m = (grid[r0 + dr:r0 + dr + h, c0 + dc:c0 + dc + w] == color)
                    if m.any() and not m.all():
                        return m

        return mask

    # ------------------------------------------------------------------
    # Apply templates to all blocks.
    # ------------------------------------------------------------------
    for r, c, h, w, color in blocks:
        new_color = remap.get(color, color)
        out[r:r + h, c:c + w] = bg

        mask = get_template_mask(color, h, w)
        if mask is not None:
            out[r:r + h, c:c + w] = np.where(mask, new_color, bg)
        else:
            # Graceful fallback: keep the block solid in its remapped color.
            out[r:r + h, c:c + w] = new_color

    return out