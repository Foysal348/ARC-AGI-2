import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Solve the ARC-style task.

    Color 5 pixels form corner brackets marking a missing object.
    The output is the missing object, inferred from nearby visible objects:
      - the 5-pixel bounding box gives the output size,
      - nearby objects give the pattern family,
      - color cycles give the color(s) of the missing object.
    """
    g = np.array(grid, dtype=int)
    if g.size == 0:
        return np.zeros_like(g)

    # If no brackets, return zeros.
    if 5 not in np.unique(g):
        return np.zeros((0, 0), dtype=int)

    # ---------------------------
    # 1. Bracket bounding box
    # ---------------------------
    ys, xs = np.where(g == 5)
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    H = r1 - r0 + 1
    W = c1 - c0 + 1

    # Work on a copy without the brackets.
    work = g.copy()
    work[work == 5] = 0

    # ---------------------------
    # Helper: connected components for a single-color mask
    # ---------------------------
    def components_single_color(color: int):
        mask = (work == color)
        seen = np.zeros_like(mask, dtype=bool)
        comps = []
        R, C = mask.shape
        for i in range(R):
            for j in range(C):
                if mask[i, j] and not seen[i, j]:
                    q = deque([(i, j)])
                    seen[i, j] = True
                    cells = []
                    minr = maxr = i
                    minc = maxc = j
                    while q:
                        y, x = q.popleft()
                        cells.append((y, x))
                        minr = min(minr, y)
                        maxr = max(maxr, y)
                        minc = min(mincc, x) if False else min(minc, x)
                        maxc = max(maxc, x)
                        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < R and 0 <= nx < C and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                    sub = np.zeros((maxr - minr + 1, maxc - minc + 1), dtype=int)
                    for yy, xx in cells:
                        sub[yy - minr, xx - minc] = color
                    comps.append({
                        "color": int(color),
                        "r0": minr, "r1": maxr,
                        "c0": minc, "c1": maxc,
                        "mask": sub,
                    })
        return comps

    def all_color_components():
        comps = []
        for col in np.unique(work):
            if col == 0:
                continue
            comps.extend(components_single_color(int(col)))
        return comps

    # ---------------------------
    # Helper: pattern detectors
    # ---------------------------
    def fits_comb(mask: np.ndarray) -> bool:
        if mask.size == 0:
            return False
        h, w = mask.shape
        if h < 3 or w < 2:
            return False
        # left column fully filled
        if not np.all(mask[:, 0] > 0):
            return False
        # full rows at h-3 and h-1
        if not np.all(mask[h-3] > 0) or not np.all(mask[h-1] > 0):
            return False
        # other rows only left column
        for i in range(h):
            if i == h-3 or i == h-1:
                continue
            if np.any(mask[i, 1:] > 0):
                return False
        return True

    def fits_ladder(mask: np.ndarray) -> bool:
        if mask.size == 0:
            return False
        h, w = mask.shape
        if w != 3 or h < 3:
            return False
        for i in range(h):
            if i % 2 == 0:
                if not np.all(mask[i] > 0):
                    return False
            else:
                if not (mask[i, 0] > 0 and mask[i, 2] > 0 and mask[i, 1] == 0):
                    return False
        return True

    def fits_hladder(mask: np.ndarray) -> bool:
        if mask.size == 0:
            return False
        h, w = mask.shape
        if h != 3 or w < 3:
            return False
        for j in range(w):
            if j % 2 == 0:
                if not np.all(mask[:, j] > 0):
                    return False
            else:
                if not (mask[0, j] > 0 and mask[2, j] > 0 and mask[1, j] == 0):
                    return False
        return True

    def find_cycle(seq):
        """Shortest prefix that repeats to cover the sequence, if possible."""
        if len(seq) <= 1:
            return list(seq)
        L = len(seq)
        for p in range(1, L // 2 + 1):
            ok = True
            for i in range(L):
                if seq[i] != seq[i % p]:
                    ok = False
                    break
            if ok:
                return seq[:p]
        return list(seq)

    def first_occurrence_order(seq):
        out = []
        seen = set()
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def most_common_color():
        nz = work[work != 0]
        if nz.size == 0:
            return 0
        vals, cnts = np.unique(nz, return_counts=True)
        return int(vals[np.argmax(cnts)])

    # ---------------------------
    # Helper: infer single color from a 1D sequence
    # ---------------------------
    def infer_color(style, axis, coord):
        comps = all_color_components()
        rel = []
        for comp in comps:
            m = comp["mask"]
            if style == "comb" and not fits_comb(m):
                continue
            if style == "ladder" and not fits_ladder(m):
                continue
            if style == "hladder" and not fits_hladder(m):
                continue
            pos = comp["c0"] if axis == "x" else comp["r0"]
            rel.append((pos, comp["color"]))

        if not rel:
            return most_common_color()

        rel.sort(key=lambda z: z[0])
        coords = [z[0] for z in rel]
        colors = [z[1] for z in rel]

        # Determine index of the marker in the sequence.
        if coord < coords[0]:
            index = -1
        elif coord > coords[-1]:
            index = len(colors)
        else:
            index = len(colors)

        # If positions are regular, extrapolate over gaps.
        if len(coords) >= 2:
            diffs = [coords[i+1] - coords[i] for i in range(len(coords)-1)]
            if all(d == diffs[0] for d in diffs) and diffs[0] > 0:
                spacing = diffs[0]
                index = int(round((coord - coords[0]) / spacing))

        cycle = first_occurrence_order(colors)
        if not cycle:
            return most_common_color()
        return cycle[index % len(cycle)]

    # ---------------------------
    # Helper: find a W-wide vertical strip nearby
    # ---------------------------
    def find_vertical_template():
        R, C = work.shape
        best = None

        def consider(start):
            nonlocal best
            if start < 0 or start + W > C:
                return
            strip = work[:, start:start+W]
            nz_cols = np.where((strip != 0).any(axis=0))[0]
            if nz_cols.size == 0:
                return
            if nz_cols.max() - nz_cols.min() + 1 < max(1, W - 1):
                return

            wr0 = max(0, r0 - H)
            wr1 = min(R, r1 + H + 1)
            count = int((strip[wr0:wr1] != 0).sum())
            if count < 2:
                return

            dist = min(abs(c0 - (start + W)), abs(start - (c1 + 1)))
            nz_rows = np.where((strip != 0).any(axis=1))[0]
            align_bonus = 0
            if nz_rows.size:
                if nz_rows.min() == r0 or nz_rows.max() == r1:
                    align_bonus += 6
                if r0 <= nz_rows.min() <= r1 or r0 <= nz_rows.max() <= r1:
                    align_bonus += 2

            score = count * 10 - dist + align_bonus
            if best is None or score > best[0]:
                best = (score, start)

        # Search left and right.
        for start in range(max(0, c0 - W - 1), -1, -1):
            consider(start)
        for start in range(min(C - W, c1 + 1), C - W + 1):
            consider(start)

        return best[1] if best is not None else None

    # ---------------------------
    # Helper: block-stack renderer (width 2)
    # ---------------------------
    def render_block_stack(strip, out_h, top_row):
        R, C = work.shape
        out = np.zeros((out_h, 2), dtype=int)

        # Snap to a full row near the requested top.
        r_start = top_row
        full_rows = []
        for rr in range(max(0, top_row - 3), min(R, top_row + 4)):
            if (strip[rr] != 0).all():
                full_rows.append(rr)
        if full_rows:
            r_start = min(full_rows)

        # Infer main color cycle from visible 2x2 blocks.
        mains = []
        max_blocks = min(out_h // 2, max(0, (R - r_start) // 2))
        for b in range(max_blocks):
            blk = strip[r_start + 2*b : r_start + 2*b + 2]
            nz = blk[blk != 0]
            if nz.size == 0:
                break
            vals, cnts = np.unique(nz, return_counts=True)
            mains.append(int(vals[np.argmax(cnts)]))
            # If the visible block is already truncated by background, stop reading.
            if (blk == 0).any():
                break

        cycle = find_cycle(mains) if mains else [most_common_color()]

        total_blocks = out_h // 2
        for idx in range(total_blocks):
            main = cycle[idx % len(cycle)]
            out[2*idx:2*idx+2, :] = main

            # Last block of the object ends with a background notch.
            if idx == total_blocks - 1:
                odd = 0
            else:
                odd = cycle[(idx + 1) % len(cycle)]

            # Notch alternates bottom-left, bottom-right, bottom-left, ...
            notch_r = 2*idx + 1
            notch_c = 0 if idx % 2 == 0 else 1
            out[notch_r, notch_c] = odd

        return out

    # ---------------------------
    # Helper: generic fallback renderer
    # ---------------------------
    def generic_fallback():
        out = np.zeros((H, W), dtype=int)
        comps = all_color_components()
        if not comps:
            return out

        # Choose the nearest component.
        best = None
        for comp in comps:
            dr = max(0, max(comp["r0"], r0) - min(comp["r1"], r1) + 1)
            dc = max(0, max(comp["c0"], c0) - min(comp["c1"], c1) + 1)
            dist_r = 0
            if comp["r1"] < r0:
                dist_r = r0 - comp["r1"]
            elif comp["r0"] > r1:
                dist_r = comp["r0"] - r1
            dist_c = 0
            if comp["c1"] < c0:
                dist_c = c0 - comp["c1"]
            elif comp["c0"] > c1:
                dist_c = comp["c0"] - c1
            score = dr + dc + dist_r + dist_c
            if best is None or score < best[0]:
                best = (score, comp)

        comp = best[1]
        m = comp["mask"]
        nz = m[m != 0]
        if nz.size == 0:
            return out

        # If single color, recolor with inferred sequence color when possible.
        single_color = (len(np.unique(nz)) == 1)
        fill_color = comp["color"]

        # Copy/shift the component into the output as much as possible.
        # Vertical alignment:
        if r1 < comp["r0"]:
            row_start = comp["r0"]
        elif r0 > comp["r1"]:
            row_start = comp["r1"] - H + 1
        else:
            row_start = r0 - (comp["r0"] - 0)

        # Horizontal alignment:
        if c1 < comp["c0"]:
            col_start = comp["c0"] - W + 1
        elif c0 > comp["c1"]:
            col_start = comp["c0"]
        else:
            col_start = c0 - (comp["c0"] - 0)

        for i in range(H):
            sr = row_start + i
            if 0 <= sr < m.shape[0]:
                for j in range(W):
                    sc = col_start + j
                    if 0 <= sc < m.shape[1]:
                        out[i, j] = m[sr, sc]

        # If the shift produced nothing, use a simple canonical shape.
        if (out == 0).all():
            if single_color:
                # Infer color from a row or column sequence.
                if H <= W:
                    fill_color = infer_color(None, "x", c0)
                else:
                    fill_color = infer_color(None, "y", r0)

                if H >= 3:
                    out[:, 0] = fill_color
                    out[H-3, :] = fill_color
                    out[H-1, :] = fill_color
                else:
                    out[:, :] = fill_color
            else:
                out[:, :] = most_common_color()
                if H > 1 and W > 1:
                    out[1:-1, 1:-1] = 0

        return out

    # ---------------------------
    # 2. Try to use a compatible vertical template
    # ---------------------------
    start = find_vertical_template()
    if start is not None:
        strip = work[:, start:start+W]
        nz = strip[strip != 0]

        if nz.size > 0:
            strip_mask = (strip != 0)
            rows = np.where(strip_mask.any(axis=1))[0]
            if rows.size:
                crop_mask = strip_mask[rows.min():rows.max()+1]
            else:
                crop_mask = strip_mask

            unique_colors = np.unique(nz)

            # Multi-color 2-wide block stack.
            if len(unique_colors) > 1 and W == 2 and H % 2 == 0:
                return render_block_stack(strip, H, r0)

            # Single-color canonical shapes.
            if len(unique_colors) == 1:
                if fits_comb(crop_mask):
                    col = infer_color("comb", "x", c0)
                    out = np.zeros((H, W), dtype=int)
                    if H == 1:
                        out[0, :] = col
                    elif H == 2:
                        out[:, :] = col
                    else:
                        out[:, 0] = col
                        out[H-3, :] = col
                        out[H-1, :] = col
                    return out

                if fits_ladder(crop_mask):
                    col = infer_color("ladder", "x", c0)
                    out = np.zeros((H, W), dtype=int)
                    if W == 3:
                        for i in range(H):
                            if i % 2 == 0:
                                out[i, :] = col
                            else:
                                out[i, 0] = col
                                out[i, 2] = col
                    else:
                        # Fallback border-like fill if ladder width is not 3.
                        out[:, 0] = col
                        out[:, -1] = col
                        out[0, :] = col
                        out[-1, :] = col
                    return out

                if fits_hladder(crop_mask):
                    col = infer_color("hladder", "y", r0)
                    out = np.zeros((H, W), dtype=int)
                    if H == 3:
                        for j in range(W):
                            if j % 2 == 0:
                                out[:, j] = col
                            else:
                                out[0, j] = col
                                out[2, j] = col
                    else:
                        out[:, 0] = col
                        out[-1, :] = col
                    return out

    # ---------------------------
    # 3. Fallback
    # ---------------------------
    return generic_fallback()