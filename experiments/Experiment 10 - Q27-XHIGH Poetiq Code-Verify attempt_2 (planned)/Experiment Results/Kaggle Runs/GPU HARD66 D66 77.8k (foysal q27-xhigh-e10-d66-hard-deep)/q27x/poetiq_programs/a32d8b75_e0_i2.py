import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the input according to the observed ARC rule.

    The input consists of a left control strip and a right canvas, separated
    by a full-height column of 6s.

    The left strip contains:
      - a top sprite (two colours),
      - a middle binary mask,
      - a bottom key made from 4 and 7 pixels.

    The right canvas is copied to the output, then the sprite is colour-swapped
    and stamped over the canvas. Each mask pixel becomes one sprite tile.

    The bottom key selects:
      - rotation of the mask:
          top-left     -> 0
          top-right    -> 90  (counterclockwise in np.rot90)
          bottom-right -> 180
          bottom-left  -> 270 (counterclockwise in np.rot90)
      - alignment corner:
          the same quadrant as the 4 marker.

    The implementation is defensive and falls back to reasonable defaults if
    some marker is missing or ambiguous.
    """
    if grid is None:
        return np.zeros((0, 0), dtype=np.uint8)

    grid = np.asarray(grid, dtype=np.uint8)
    if grid.size == 0:
        return grid.copy()

    H, W = grid.shape

    # ------------------------------------------------------------------
    # 1. Locate the main vertical separator column of 6s.
    # ------------------------------------------------------------------
    def separator_score(col: int) -> int:
        return int(np.sum(grid[:, col] == 6))

    sep_col = None
    for c in range(W):
        if separator_score(c) >= max(1, int(0.95 * H)):
            sep_col = c
            break

    if sep_col is None:
        # Fallback: choose the column with the strongest 6-line.
        scores = [separator_score(c) for c in range(W)]
        best = max(scores)
        if best >= max(1, H // 2):
            sep_col = scores.index(best)

    if sep_col is None:
        return grid.copy()

    left = grid[:, :sep_col].copy()
    canvas = grid[:, sep_col + 1:].copy()
    out = canvas.copy()
    CH, CW = out.shape
    LH, LW = left.shape

    if CH == 0 or CW == 0:
        return out

    # ------------------------------------------------------------------
    # 2. Connected components in the left strip.
    #    We consider non-zero and non-6 cells.
    # ------------------------------------------------------------------
    def is_content(r: int, c: int) -> bool:
        if 0 <= r < LH and 0 <= c < LW:
            v = int(left[r, c])
            return v not in (0, 6)
        return False

    visited = np.zeros((LH, LW), dtype=bool)
    comps = []

    for r in range(LH):
        for c in range(LW):
            if not visited[r, c] and is_content(r, c):
                q = deque([(r, c)])
                visited[r, c] = True
                cells = [(r, c)]
                while q:
                    rr, cc = q.popleft()
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < LH and 0 <= nc < LW and not visited[nr, nc] and is_content(nr, nc):
                                visited[nr, nc] = True
                                cells.append((nr, nc))
                                q.append((nr, nc))
                comps.append(cells)

    if not comps:
        return out

    # Top sprite: component with smallest top row, then leftmost.
    comps_sorted = sorted(comps, key=lambda cells: (min(r for r, c in cells), min(c for r, c in cells)))
    sprite_cells = comps_sorted[0]

    sr0 = min(r for r, c in sprite_cells)
    sr1 = max(r for r, c in sprite_cells)
    sc0 = min(c for r, c in sprite_cells)
    sc1 = max(c for r, c in sprite_cells)
    sprite = left[sr0:sr1 + 1, sc0:sc1 + 1].copy()

    # Swap the two non-background colours in the sprite.
    sprite_colors = [int(v) for v in np.unique(sprite) if int(v) not in (0, 6)]
    if len(sprite_colors) == 2:
        a, b = sprite_colors
        sprite = np.where(sprite == a, b, np.where(sprite == b, a, sprite))
    else:
        # If the sprite is monochrome or unexpectedly complex, leave it unchanged.
        sprite = sprite.copy()

    sh, sw = sprite.shape
    if sh == 0 or sw == 0:
        return out

    # ------------------------------------------------------------------
    # 3. Extract the main mask: the largest component that is not the sprite
    #    and not an obvious small bottom-key component.
    # ------------------------------------------------------------------
    def comp_values(cells):
        vals = set()
        for r, c in cells:
            v = int(left[r, c])
            if v not in (0, 6):
                vals.add(v)
        return vals

    def is_bottom_key_like(cells):
        vals = comp_values(cells)
        # Small indicators made only of 4/7, or tiny isolated components near bottom.
        if vals <= {4, 7} and len(cells) <= 8:
            # Strongly prefer treating it as key if it is in the lower part.
            if max(r for r, c in cells) >= LH * 0.6:
                return True
        return False

    candidates = []
    for cells in comps[1:]:
        if not is_bottom_key_like(cells):
            candidates.append(cells)

    if not candidates:
        # Fallback: use the largest non-sprite component.
        candidates = comps[1:]

    if not candidates:
        return out

    # Largest candidate is the mask.
    mask_cells = max(candidates, key=len)

    mr0 = min(r for r, c in mask_cells)
    mr1 = max(r for r, c in mask_cells)
    mc0 = min(c for r, c in mask_cells)
    mc1 = max(c for r, c in mask_cells)

    mask = np.zeros((mr1 - mr0 + 1, mc1 - mc0 + 1), dtype=bool)
    for r, c in mask_cells:
        mask[r - mr0, c - mc0] = True

    if not mask.any():
        return out

    # ------------------------------------------------------------------
    # 4. Decode the bottom key:
    #    - 4 marker gives the alignment quadrant / rotation quadrant.
    #    - 7 marker is used as a sanity hint for horizontal/vertical orientation.
    # ------------------------------------------------------------------
    # Indicator cells are remaining 4/7 pixels not belonging to sprite or mask.
    sprite_set = set(sprite_cells)
    mask_set = set(mask_cells)

    fours = []
    sevens = []

    for r in range(LH):
        for c in range(LW):
            if (r, c) in sprite_set or (r, c) in mask_set:
                continue
            v = int(left[r, c])
            if v == 4:
                fours.append((r, c))
            elif v == 7:
                sevens.append((r, c))

    # If no 4 is found, try to use the lowest 7/4 component as a rough key.
    if not fours:
        # Use any remaining small component as fallback.
        for r in range(LH):
            for c in range(LW):
                if (r, c) in sprite_set or (r, c) in mask_set:
                    continue
                if left[r, c] in (4, 7):
                    if left[r, c] == 4:
                        fours.append((r, c))
                    else:
                        sevens.append((r, c))

    # Default: top-left.
    quadrant = "tl"

    if fours:
        fr, fc = sum(r for r, c in fours) // len(fours), sum(c for r, c in fours) // len(fours)

        # Use the bounding box of all indicator pixels if available.
        key_pixels = fours + sevens
        if key_pixels:
            min_r = min(r for r, c in key_pixels)
            max_r = max(r for r, c in key_pixels)
            min_c = min(c for r, c in key_pixels)
            max_c = max(c for r, c in key_pixels)

            span_r = max(1, max_r - min_r)
            span_c = max(1, max_c - min_c)

            # Classify the 4 position inside the key bounding box.
            # A small epsilon makes exact corners robust.
            top = fr <= min_r + 0.45 * span_r
            bottom = fr >= max_r - 0.45 * span_r
            leftq = fc <= min_c + 0.45 * span_c
            rightq = fc >= max_c - 0.45 * span_c

            # If both top and bottom are true (tiny box), prefer top.
            # If both left and right are true (tiny box), prefer left.
            if top and not bottom:
                vside = "top"
            elif bottom and not top:
                vside = "bottom"
            else:
                vside = "top" if fr < (min_r + max_r) / 2 else "bottom"

            if leftq and not rightq:
                hside = "left"
            elif rightq and not leftq:
                hside = "right"
            else:
                hside = "left" if fc < (min_c + max_c) / 2 else "right"

            if vside == "top":
                quadrant = "tl" if hside == "left" else "tr"
            else:
                quadrant = "bl" if hside == "left" else "br"

    # If no 7 marker is found, quadrant still works as a fallback.
    # Rotation mapping from quadrant:
    #   tl -> 0, tr -> 90 ccw, br -> 180, bl -> 270 ccw.
    rot_map = {
        "tl": 0,
        "tr": 1,
        "br": 2,
        "bl": 3,
    }
    rot_k = rot_map[quadrant]

    # ------------------------------------------------------------------
    # 5. Rotate the mask and choose alignment offsets.
    # ------------------------------------------------------------------
    mask_rot = np.rot90(mask, k=rot_k)

    mh, mw = mask_rot.shape
    stamp_h = mh * sh
    stamp_w = mw * sw

    # Alignment corner:
    if quadrant == "tl":
        row_off = 0
        col_off = 0
    elif quadrant == "tr":
        row_off = 0
        col_off = CW - stamp_w
    elif quadrant == "bl":
        row_off = CH - stamp_h
        col_off = 0
    else:  # br
        row_off = CH - stamp_h
        col_off = CW - stamp_w

    # Keep offsets in range.
    row_off = int(max(0, min(CH, row_off)))
    col_off = int(max(0, min(CW, col_off)))

    # ------------------------------------------------------------------
    # 6. Stamp the sprite tiles according to the mask.
    # ------------------------------------------------------------------
    for i in range(mh):
        for j in range(mw):
            if not mask_rot[i, j]:
                continue

            r0 = row_off + i * sh
            c0 = col_off + j * sw
            r1 = r0 + sh
            c1 = c0 + sw

            if r0 >= CH or c0 >= CW:
                continue

            r0c, c0c = max(0, r0), max(0, c0)
            r1c, c1c = min(CH, r1), min(CW, c1)

            # If the tile is partially outside the canvas, stamp only the visible part.
            sr0 = r0c - r0
            sc0 = c0c - c0
            sr1 = sr0 + (r1c - r0c)
            sc1 = sc0 + (c1c - c0c)

            tile_piece = sprite[sr0:sr1, sc0:sc1]
            target = out[r0c:r1c, c0c:c1c]

            # If the sprite has 0 pixels, treat them as transparent.
            if np.any(tile_piece == 0):
                mask_nonzero = tile_piece != 0
                target[mask_nonzero] = tile_piece[mask_nonzero]
            else:
                target[...] = tile_piece

    return out