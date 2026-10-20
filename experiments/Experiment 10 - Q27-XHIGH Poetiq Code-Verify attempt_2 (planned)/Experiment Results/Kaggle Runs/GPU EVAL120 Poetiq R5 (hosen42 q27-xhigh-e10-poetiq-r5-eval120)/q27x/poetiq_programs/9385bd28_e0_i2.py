import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Complete rectangles whose corners are marked by small L-shaped pieces.

    Rule:
      - The background color is the most frequent color.
      - A bottom-left legend gives pairs: corner_color -> fill_color.
      - Non-background, non-legend cells are grouped into components.
        A valid corner is a 3-cell L-shape inside a 2x2 box.
        A valid degenerate rectangle is a full 2x2 block.
      - Two opposite L-corners of the same color define the bounding box
        of a rectangle. Its interior/bounding box is painted with the
        legend fill color, then its corner cells are redrawn in the
        corner color.
      - Rectangles are painted from largest area to smallest area, so
        smaller rectangles remain visible over larger ones.

    The important corner orientation is:
      TL: top-left, top-right, bottom-left
      TR: top-left, top-right, bottom-right
      BL: bottom-left, bottom-right, top-left
      BR: top-right, bottom-left, bottom-right
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        return grid.copy()

    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    out = grid.copy()

    # -------------------------------------------------------------------------
    # 1. Find the background color.
    # -------------------------------------------------------------------------
    vals, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    ties = vals[counts == max_count]

    if len(ties) == 1:
        background = int(ties[0])
    else:
        # Tie-break by border frequency, then by preference for 0, then smaller.
        border = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        best = int(ties[0])
        best_key = None
        for v in ties:
            v = int(v)
            border_count = int(np.sum(border == v))
            key = (border_count, 1 if v == 0 else 0, -v)
            if best_key is None or key > best_key:
                best_key = key
                best = v
        background = best

    # -------------------------------------------------------------------------
    # 2. Extract the bottom-left legend.
    #    It is a vertical run of pairs: left cell = corner color,
    #    right cell = fill color.
    # -------------------------------------------------------------------------
    legend_mask = np.zeros((H, W), dtype=bool)
    mapping = {}

    if H >= 2 and W >= 2:
        candidates = []
        max_c = min(W - 2, max(0, W // 2))

        for c in range(max_c + 1):
            start = None
            for r in range(H // 2, H):
                if grid[r, c] != background:
                    if start is None:
                        start = r
                else:
                    if start is not None:
                        end = r - 1
                        length = end - start + 1
                        distinct = int(np.unique(grid[start:end + 1, c]).size)
                        pair_count = int(np.sum(grid[start:end + 1, c + 1] != background))
                        candidates.append({
                            "distinct": distinct,
                            "length": length,
                            "end": end,
                            "start": start,
                            "c": c,
                            "r0": start,
                            "r1": end,
                            "pair_count": pair_count,
                        })
                        start = None

            if start is not None:
                end = H - 1
                length = end - start + 1
                distinct = int(np.unique(grid[start:end + 1, c]).size)
                pair_count = int(np.sum(grid[start:end + 1, c + 1] != background))
                candidates.append({
                    "distinct": distinct,
                    "length": length,
                    "end": end,
                    "start": start,
                    "c": c,
                    "r0": start,
                    "r1": end,
                    "pair_count": pair_count,
                })

        if candidates:
            # The right column of the legend often looks like another vertical run.
            # Prefer the left column: if c-1 has the exact same run, skip c.
            runs_set = {(x["c"], x["r0"], x["r1"]) for x in candidates}
            pool = []
            for x in candidates:
                if x["c"] > 0 and (x["c"] - 1, x["r0"], x["r1"]) in runs_set:
                    continue
                pool.append(x)

            if not pool:
                pool = candidates

            # Multi-entry legends are the safest choice.
            multi = [x for x in pool if x["distinct"] >= 2 and x["length"] >= 2]
            if multi:
                best = max(
                    multi,
                    key=lambda x: (
                        x["distinct"],
                        x["length"],
                        -x["c"],
                        x["pair_count"],
                        x["end"],
                        x["start"],
                    ),
                )
            else:
                # Single-entry legend: prefer a true horizontal pair, then bottom-most,
                # then left-most.
                best = max(
                    pool,
                    key=lambda x: (
                        x["pair_count"],
                        x["end"],
                        -x["c"],
                        x["distinct"],
                        x["length"],
                    ),
                )

            c = best["c"]
            r0 = best["r0"]
            r1 = best["r1"]

            for r in range(r0, r1 + 1):
                legend_mask[r, c] = True
                legend_mask[r, c + 1] = True
                src = int(grid[r, c])
                fill = int(grid[r, c + 1])
                if src != background:
                    mapping[src] = fill

    # -------------------------------------------------------------------------
    # 3. Find small non-background components:
    #      - size 3 in a 2x2 box -> L corner
    #      - size 4 in a 2x2 box -> degenerate rectangle
    # -------------------------------------------------------------------------
    visited = np.zeros((H, W), dtype=bool)
    l_components = {}
    block_components = {}
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if visited[r, c] or legend_mask[r, c] or grid[r, c] == background:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells = []

            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))

                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < H
                        and 0 <= nc < W
                        and not visited[nr, nc]
                        and not legend_mask[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if len(cells) not in (3, 4):
                continue

            rmin = min(x[0] for x in cells)
            rmax = max(x[0] for x in cells)
            cmin = min(x[1] for x in cells)
            cmax = max(x[1] for x in cells)

            if rmax - rmin != 1 or cmax - cmin != 1:
                continue

            if len(cells) == 3:
                cell_set = set(cells)
                missing = None
                for rr in (rmin, rmax):
                    for cc in (cmin, cmax):
                        if (rr, cc) not in cell_set:
                            missing = (rr, cc)
                            break
                    if missing is not None:
                        break

                if missing == (rmax, cmax):
                    orientation = "TL"
                elif missing == (rmax, cmin):
                    orientation = "TR"
                elif missing == (rmin, cmax):
                    orientation = "BL"
                elif missing == (rmin, cmin):
                    orientation = "BR"
                else:
                    continue

                l_components.setdefault(color, []).append({
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                    "orientation": orientation,
                })

            else:
                block_components.setdefault(color, []).append({
                    "rmin": rmin,
                    "rmax": rmax,
                    "cmin": cmin,
                    "cmax": cmax,
                })

    # -------------------------------------------------------------------------
    # 4. Build rectangles from legend colors.
    # -------------------------------------------------------------------------
    def make_l_rectangle(a, b, src, fill):
        """Make a rectangle from two opposite L-corners, if valid."""
        oa, ob = a["orientation"], b["orientation"]

        if oa == "TL" and ob == "BR":
            tl, br = a, b
            kind = "TLBR"
        elif oa == "BR" and ob == "TL":
            tl, br = b, a
            kind = "TLBR"
        elif oa == "TR" and ob == "BL":
            tr, bl = a, b
            kind = "TRBL"
        elif oa == "BL" and ob == "TR":
            tr, bl = b, a
            kind = "TRBL"
        else:
            return None

        rmin = min(a["rmin"], b["rmin"])
        rmax = max(a["rmax"], b["rmax"])
        cmin = min(a["cmin"], b["cmin"])
        cmax = max(a["cmax"], b["cmax"])

        if rmax - rmin < 1 or cmax - cmin < 1:
            return None

        if kind == "TLBR":
            if (
                tl["rmin"] != rmin
                or tl["cmin"] != cmin
                or br["rmax"] != rmax
                or br["cmax"] != cmax
            ):
                return None
            orientations = ["TL", "BR"]
        else:
            if (
                tr["rmin"] != rmin
                or tr["cmax"] != cmax
                or bl["rmax"] != rmax
                or bl["cmin"] != cmin
            ):
                return None
            orientations = ["TR", "BL"]

        return {
            "src": src,
            "fill": fill,
            "rmin": rmin,
            "rmax": rmax,
            "cmin": cmin,
            "cmax": cmax,
            "degenerate": False,
            "orientations": orientations,
        }

    rectangles = []

    for src, fill in mapping.items():
        src = int(src)
        fill = int(fill)
        if src == background:
            continue

        ls = l_components.get(src, [])
        blks = block_components.get(src, [])

        chosen = None

        # Prefer a valid pair of opposite L-corners.
        if len(ls) >= 2:
            best_area = -1
            for i in range(len(ls)):
                for j in range(i + 1, len(ls)):
                    cand = make_l_rectangle(ls[i], ls[j], src, fill)
                    if cand is not None:
                        area = (cand["rmax"] - cand["rmin"] + 1) * (cand["cmax"] - cand["cmin"] + 1)
                        if area > best_area:
                            best_area = area
                            chosen = cand

            if chosen is not None:
                rectangles.append(chosen)

        # Otherwise, a full 2x2 block is a degenerate rectangle.
        if chosen is None and blks:
            for bl in blks:
                rectangles.append({
                    "src": src,
                    "fill": fill,
                    "rmin": bl["rmin"],
                    "rmax": bl["rmax"],
                    "cmin": bl["cmin"],
                    "cmax": bl["cmax"],
                    "degenerate": True,
                    "orientations": [],
                })

    if not rectangles:
        return out

    # -------------------------------------------------------------------------
    # 5. Paint rectangles from largest to smallest.
    # -------------------------------------------------------------------------
    def rect_area(rect):
        return (rect["rmax"] - rect["rmin"] + 1) * (rect["cmax"] - rect["cmin"] + 1)

    rectangles.sort(key=lambda r: (rect_area(r), r["src"]), reverse=True)

    def bracket_cells(rmin, rmax, cmin, cmax, orientation):
        if orientation == "TL":
            return [(rmin, cmin), (rmin, cmin + 1), (rmin + 1, cmin)]
        if orientation == "TR":
            return [(rmin, cmax - 1), (rmin, cmax), (rmin + 1, cmax)]
        if orientation == "BL":
            return [(rmax - 1, cmin), (rmax, cmin), (rmax, cmin + 1)]
        if orientation == "BR":
            return [(rmax - 1, cmax), (rmax, cmax - 1), (rmax, cmax)]
        return []

    for rect in rectangles:
        rmin = rect["rmin"]
        rmax = rect["rmax"]
        cmin = rect["cmin"]
        cmax = rect["cmax"]

        if rmin > rmax or cmin > cmax:
            continue

        # Fill the bounding box, preserving legend cells.
        region = out[rmin:rmax + 1, cmin:cmax + 1]
        legend_region = legend_mask[rmin:rmax + 1, cmin:cmax + 1]

        if legend_region.any():
            region[~legend_region] = rect["fill"]
        else:
            region[:] = rect["fill"]

        # Redraw canonical corner pieces for non-degenerate rectangles.
        if not rect["degenerate"]:
            for orientation in rect["orientations"]:
                for rr, cc in bracket_cells(rmin, rmax, cmin, cmax, orientation):
                    if 0 <= rr < H and 0 <= cc < W and not legend_mask[rr, cc]:
                        out[rr, cc] = rect["src"]

    return out