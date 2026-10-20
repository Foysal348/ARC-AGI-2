import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    ARC-style transformation.

    Observed rule
    ------------
    - The input is split by a full-height column of 6s into a left "control strip"
      and a right canvas.  Any additional full-height 6 columns inside the canvas
      are also removed.
    - The left control strip contains:
        * a top small colored pattern (the "sprite"),
        * a larger binary mask below it (the largest connected non-0, non-6 region),
        * a small bottom indicator made of 4 and/or 7 pixels that selects the
          mask orientation.
    - The sprite colors are swapped and the selected rotated mask is stamped
      onto the right canvas: every mask cell becomes one full sprite tile.
    - The sprite size is inferred from the top sprite's bounding box.
    - The mask orientation is inferred from the bottom indicator:
        7 horizontal + 4 present       -> rotate 90 degrees
        7 horizontal + no 4            -> rotate 270 degrees
        7 vertical   + 4 present       -> rotate 180 degrees
        7 vertical   + no 4            -> rotate 0 degrees
    - If the indicator is missing, choose the first rotation that fits the canvas
      in the preferred order: 0, 180, 90, 270.
    - The stamped mask is aligned to the top/right of the canvas by default;
      if a rotated mask is wider than the canvas, it is left/top aligned.

    The implementation is deliberately defensive and tries to follow the
    generator-like structure of the provided examples.
    """

    if grid is None:
        return np.zeros((0, 0), dtype=np.uint8)

    grid = np.asarray(grid)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1. Find the main vertical separator column (first full/mostly-6 column)
    # ------------------------------------------------------------------
    def col_is_separator(col):
        if col < 0 or col >= W:
            return False
        # Require a strong vertical line of 6s.
        return np.mean(grid[:, col] == 6) > 0.85

    sep_cols = [c for c in range(W) if col_is_separator(c)]
    if not sep_cols:
        # Fallback: first column containing many 6s.
        scores = [(np.sum(grid[:, c] == 6), c) for c in range(W)]
        scores.sort(reverse=True)
        if scores[0][0] > H * 0.5:
            sep_cols = [scores[0][1]]

    if not sep_cols:
        # No valid separator: return input unchanged.
        return grid.copy()

    main_sep = sep_cols[0]
    left_strip = grid[:, :main_sep].copy()
    canvas_cols = [c for c in range(main_sep + 1, W) if not col_is_separator(c)]
    canvas = grid[:, canvas_cols].copy()
    out = canvas.copy()
    CH, CW = out.shape

    # ------------------------------------------------------------------
    # 2. Extract top sprite from the left strip
    # ------------------------------------------------------------------
    def is_sprite_cell(x, y):
        if not (0 <= x < left_strip.shape[1] and 0 <= y < left_strip.shape[0]):
            return False
        v = int(left_strip[x, y])
        return v not in (0, 6)

    visited = np.zeros(left_strip.shape, dtype=bool)
    components = []

    for r in range(left_strip.shape[0]):
        for c in range(left_strip.shape[1]):
            if not visited[r, c] and is_sprite_cell(r, c):
                q = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                while q:
                    rr, cc = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < left_strip.shape[0] and 0 <= nc < left_strip.shape[1]:
                            if not visited[nr, nc] and is_sprite_cell(nr, nc):
                                visited[nr, nc] = True
                                cells.append((nr, nc))
                                q.append((nr, nc))
                components.append(cells)

    if not components:
        return out

    # Top sprite: component with smallest top row, then leftmost.
    components.sort(key=lambda cells: (min(r for r, c in cells), min(c for r, c in cells)))
    top_cells = components[0]
    rmin = min(r for r, c in top_cells)
    rmax = max(r for r, c in top_cells)
    cmin = min(c for r, c in top_cells)
    cmax = max(c for r, c in top_cells)

    sprite = left_strip[rmin:rmax + 1, cmin:cmax + 1].copy()

    # Swap the two non-zero colors in the sprite.
    nz = [int(v) for v in np.unique(sprite) if int(v) not in (0, 6)]
    if len(nz) == 2:
        a, b = nz
        swapped = np.where(sprite == a, b, np.where(sprite == b, a, sprite))
        sprite = swapped
    elif len(nz) == 1:
        sprite = sprite.copy()
    else:
        # If somehow no usable sprite, leave as is.
        sprite = sprite.copy()

    sh, sw = sprite.shape

    # ------------------------------------------------------------------
    # 3. Extract the main mask: largest component below the top sprite
    # ------------------------------------------------------------------
    bottom = np.zeros_like(left_strip, dtype=bool)
    remaining = []
    top_bottom = max(r for r, c in top_cells)

    for cells in components[1:]:
        cells_bottom = max(r for r, c in cells)
        # A component that is entirely below the top sprite is a mask candidate.
        if min(r for r, c in cells) > top_bottom - 1:
            remaining.append(cells)

    # If the split above did not work, take all non-top components.
    if not remaining:
        remaining = components[1:]

    # Exclude obvious bottom key components: very small and containing only 4/7,
    # or lying in a 6-framed lower area.  The main mask is usually the largest.
    def comp_value_set(cells):
        vals = set()
        for r, c in cells:
            v = int(left_strip[r, c])
            if v not in (0, 6):
                vals.add(v)
        return vals

    # Prefer a component that does not consist only of {4,7} and is reasonably large.
    preferred = []
    for cells in remaining:
        vals = comp_value_set(cells)
        is_bottom_key_like = (vals <= {4, 7}) and len(cells) <= 6
        if not is_bottom_key_like:
            preferred.append(cells)
        else:
            preferred.append(cells)  # keep as fallback

    if preferred:
        # Choose the largest non-bottom-key component.
        def score(cells):
            vals = comp_value_set(cells)
            bottom_like = (vals <= {4, 7}) and len(cells) <= 6
            # Penalize bottom key strongly.
            penalty = 1000 if bottom_like else 0
            return (len(cells) - penalty,
                    -max(r for r, c in cells))

        preferred.sort(key=score, reverse=True)
        mask_cells = preferred[0]
    else:
        mask_cells = None

    if not mask_cells:
        # No mask found: return canvas unchanged.
        return out

    # Binary mask bounding box.
    mrmin = min(r for r, c in mask_cells)
    mrmax = max(r for r, c in mask_cells)
    mcmin = min(c for r, c in mask_cells)
    mcmax = max(c for r, c in mask_cells)
    mask = np.zeros((mrmax - mrmin + 1, mcmax - mcmin + 1), dtype=bool)
    for r, c in mask_cells:
        mask[r - mrmin, c - mcmin] = True

    # ------------------------------------------------------------------
    # 4. Infer mask orientation from the bottom indicator (4 / 7 markers)
    # ------------------------------------------------------------------
    def find_7_4_orientation(strip):
        # Look in the bottom third of the left strip for 4/7 pixels.
        h, w = strip.shape
        start = int(h * 0.55)
        sevens = [(r, c) for r in range(start, h) for c in range(w) if strip[r, c] == 7]
        fours  = [(r, c) for r in range(start, h) for c in range(w) if strip[r, c] == 4]
        if not sevens:
            return None
        has4 = bool(fours)

        # Determine if the 7 marker is horizontal or vertical by nearest neighbor.
        horiz = False
        vert = False
        for r, c in sevens:
            if 0 <= c + 1 < w and strip[r, c + 1] == 7:
                horiz = True
            if 0 <= r + 1 < h and strip[r + 1, c] == 7:
                vert = True

        # If multiple 7s form both, use the dominant line length.
        if horiz and vert:
            # Count connected horizontal vs vertical pairs.
            hpairs = sum(1 for r, c in sevens if 0 <= c + 1 < w and strip[r, c + 1] == 7)
            vpairs = sum(1 for r, c in sevens if 0 <= r + 1 < h and strip[r + 1, c] == 7)
            if vpairs > hpairs:
                horiz = False
            elif hpairs > vpairs:
                vert = False
            else:
                # Tie: prefer horizontal if 4 is present (matches examples),
                # otherwise vertical.
                if has4:
                    vert = False
                else:
                    horiz = False

        if horiz and not vert:
            return 90 if has4 else 270
        if vert and not horiz:
            return 180 if has4 else 0

        # Fallback: use position of 4 if any.
        if has4:
            fr, fc = fours[0]
            # Rough heuristic: 4 in right half -> 180, left half -> 90.
            return 180 if fc >= w // 2 else 90
        return 0

    desired_rot = find_7_4_orientation(left_strip)

    # ------------------------------------------------------------------
    # 5. Choose a rotation that fits the canvas and stamp tile size
    # ------------------------------------------------------------------
    def rotate_mask(m, k):
        # k: 0, 1, 2, 3 for 0/90/180/270 clockwise
        return np.rot90(m, k=k)

    def fits(m):
        th, tw = m.shape
        if sh == 0 or sw == 0:
            return False
        return th * sh <= CH and tw * sw <= CW

    if desired_rot is None:
        candidate_rots = [0, 2, 1, 3]
    else:
        # Put desired rotation first, then plausible alternatives.
        candidate_rots = [desired_rot // 90]
        for r in [0, 2, 1, 3]:
            if r not in candidate_rots:
                candidate_rots.append(r)

    chosen_rot = None
    for k in candidate_rots:
        if fits(rotate_mask(mask, k)):
            chosen_rot = k
            break

    if chosen_rot is None:
        # Last resort: use 0 if it fits, otherwise return unchanged.
        if fits(mask):
            chosen_rot = 0
        else:
            return out

    m = rotate_mask(mask, chosen_rot)
    mh, mw = m.shape
    tile_h, tile_w = sh, sw
    ph, pw = mh * tile_h, mw * tile_w

    # Alignment:
    # - Usually top/right (examples 1 and 2).
    # - If mask width exactly equals canvas tile width, left/right both same.
    # - If rotated mask is very wide relative to canvas, left/top.
    if pw <= CW:
        row_off = 0
        col_off = CW - pw
    else:
        row_off = 0
        col_off = 0

    # Optional: if the mask has a solid bottom row and no solid top row,
    # and the canvas has extra vertical space, prefer bottom alignment.
    # This matches example 2 behavior when the mask is tall and right-aligned.
    if ph < CH and mw == (CW // tile_w):
        # right-aligned width already set; check vertical preference.
        # If mask has a full bottom row and top row is not full, bottom-align.
        full_top = np.all(m[0, :])
        full_bot = np.all(m[-1, :])
        if full_bot and not full_top:
            row_off = CH - ph

    # Apply stamp.
    for i in range(mh):
        for j in range(mw):
            if not m[i, j]:
                continue
            r0 = row_off + i * tile_h
            c0 = col_off + j * tile_w
            r1 = r0 + tile_h
            c1 = c0 + tile_w
            if r1 > CH or c1 > CW:
                continue
            out[r0:r1, c0:c1] = sprite

    return out