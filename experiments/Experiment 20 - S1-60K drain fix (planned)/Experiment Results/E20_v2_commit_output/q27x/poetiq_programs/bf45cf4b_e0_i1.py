import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by using one object as a binary mask and the other as a tile.

    The background color is inferred from the border/most frequent color.
    The mask object consists of a single non-background color. The other object
    is the tile. The output is constructed by replacing every ON cell of the mask
    with the tile and every OFF cell with a background block.
    """
    grid = np.asarray(grid, dtype=int)

    if grid.ndim != 2:
        return grid

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    def _background(g):
        """Return the background color, preferring a uniform border color."""
        corners = [g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]]
        if len(set(corners)) == 1:
            return int(corners[0])

        vals, counts = np.unique(g, return_counts=True)
        return int(vals[np.argmax(counts)])

    bg = _background(grid)
    nonbg = grid != bg

    if not nonbg.any():
        return grid.copy()

    def _components(g, background):
        """
        Find 8-connected components of non-background pixels.

        Each component is stored with its bounding box, the set of non-background
        colors in it, and its pixel count.
        """
        h, w = g.shape
        visited = np.zeros((h, w), dtype=bool)
        components = []

        for r in range(h):
            for c in range(w):
                if g[r, c] != background and not visited[r, c]:
                    stack = [(r, c)]
                    visited[r, c] = True

                    min_r = max_r = r
                    min_c = max_c = c
                    colors = set()
                    count = 0

                    while stack:
                        cr, cc = stack.pop()
                        value = int(g[cr, cc])
                        colors.add(value)
                        count += 1

                        if cr < min_r:
                            min_r = cr
                        if cr > max_r:
                            max_r = cr
                        if cc < min_c:
                            min_c = cc
                        if cc > max_c:
                            max_c = cc

                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue

                                nr, nc = cr + dr, cc + dc
                                if (
                                    0 <= nr < h
                                    and 0 <= nc < w
                                    and not visited[nr, nc]
                                    and g[nr, nc] != background
                                ):
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))

                    components.append({
                        "bbox": (min_r, max_r, min_c, max_c),
                        "colors": colors,
                        "count": count,
                    })

        return components

    def _make_output(mask_bbox, mask_color, tile_bbox):
        """Stamp the tile according to the binary mask."""
        mr0, mr1, mc0, mc1 = mask_bbox
        tr0, tr1, tc0, tc1 = tile_bbox

        if mask_color is None:
            mask_patch = (grid[mr0:mr1 + 1, mc0:mc1 + 1] != bg)
        else:
            mask_patch = (grid[mr0:mr1 + 1, mc0:mc1 + 1] == mask_color)

        # Safety fallback if the selected color did not mark any mask cell.
        if not mask_patch.any():
            mask_patch = (grid[mr0:mr1 + 1, mc0:mc1 + 1] != bg)

        tile = grid[tr0:tr1 + 1, tc0:tc1 + 1]
        th, tw = tile.shape
        mh, mw = mask_patch.shape

        out = np.full((mh * th, mw * tw), bg, dtype=int)

        for i in range(mh):
            for j in range(mw):
                if mask_patch[i, j]:
                    out[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tile

        return out

    def _select_mask_color(g, background):
        """
        Fallback selector for the mask color.

        Prefer a non-background color whose bounding box contains no other
        non-background colors and which is as far as possible from other colors.
        """
        nonbg_mask = g != background
        colors = np.unique(g[nonbg_mask])

        if colors.size == 0:
            return None, None

        infos = []

        for color in colors:
            color = int(color)
            rows, cols = np.where(g == color)

            min_r = int(rows.min())
            max_r = int(rows.max())
            min_c = int(cols.min())
            max_c = int(cols.max())

            sub = g[min_r:max_r + 1, min_c:max_c + 1]
            sub_nonbg = sub[sub != background]
            only = bool(len(np.unique(sub_nonbg)) == 1)

            count = int(rows.size)
            area = (max_r - min_r + 1) * (max_c - min_c + 1)
            fill = count / area if area > 0 else 0.0

            infos.append([color, min_r, max_r, min_c, max_c, only, count, fill])

        candidates = [info for info in infos if info[5]]
        if not candidates:
            candidates = infos

        best = None
        best_key = None

        for info in candidates:
            color, min_r, max_r, min_c, max_c, only, count, fill = info

            other = nonbg_mask & (g != color)
            if other.any():
                rr, cc = np.where(other)
                dr = np.maximum(0, np.maximum(min_r - rr, rr - max_r))
                dc = np.maximum(0, np.maximum(min_c - cc, cc - max_c))
                dist = int(np.maximum(dr, dc).min())
            else:
                dist = 10**9

            # Prefer:
            # 1) a clean single-color bounding box,
            # 2) larger distance from other objects,
            # 3) lower fill ratio,
            # 4) fewer pixels.
            key = (0 if only else 1, -dist, fill, count)

            if best_key is None or key < best_key:
                best_key = key
                best = info

        if best is None:
            return None, None

        return best[0], (best[1], best[2], best[3], best[4])

    # Fast path: usually there are exactly two 8-connected objects.
    components = _components(grid, bg)

    if len(components) == 2:
        comp_a, comp_b = components
        a_single = len(comp_a["colors"]) == 1
        b_single = len(comp_b["colors"]) == 1

        if a_single and not b_single:
            return _make_output(
                comp_a["bbox"],
                next(iter(comp_a["colors"])),
                comp_b["bbox"],
            )

        if b_single and not a_single:
            return _make_output(
                comp_b["bbox"],
                next(iter(comp_b["colors"])),
                comp_a["bbox"],
            )

    # Fallback for disconnected objects or ambiguous components.
    mask_color, mask_bbox = _select_mask_color(grid, bg)

    if mask_color is None:
        return grid.copy()

    mr0, mr1, mc0, mc1 = mask_bbox

    # Tile is everything non-background outside the mask's bounding box.
    tile_mask = nonbg.copy()
    tile_mask[mr0:mr1 + 1, mc0:mc1 + 1] = False

    if not tile_mask.any():
        return grid.copy()

    rr, cc = np.where(tile_mask)
    tile_bbox = (
        int(rr.min()),
        int(rr.max()),
        int(cc.min()),
        int(cc.max()),
    )

    return _make_output(mask_bbox, mask_color, tile_bbox)