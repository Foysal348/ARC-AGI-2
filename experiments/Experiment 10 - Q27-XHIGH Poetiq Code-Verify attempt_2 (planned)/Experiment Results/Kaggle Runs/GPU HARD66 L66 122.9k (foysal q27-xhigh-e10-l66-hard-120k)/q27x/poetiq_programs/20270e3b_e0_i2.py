import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move the smaller 4-colored object marked by 7 pixels so that its marked edge
    is placed adjacent to the marked edge of the other object. Remove all 7 pixels
    and crop the result to the bounding box of all remaining 4 pixels.
    """
    a = np.array(grid, dtype=int)
    if a.size == 0:
        return a.copy()
    if a.ndim != 2:
        return a.copy()

    H, W = a.shape
    dirs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def crop_to_4(g: np.ndarray) -> np.ndarray:
        """Crop to the bounding box of all 4 pixels."""
        if not np.any(g == 4):
            return g.copy()
        rows = np.where((g == 4).any(axis=1))[0]
        cols = np.where((g == 4).any(axis=0))[0]
        return g[rows.min():rows.max() + 1, cols.min():cols.max() + 1].copy()

    # Background is the most common color that is neither object (4) nor marker (7).
    if np.any(a != 7):
        vals, counts = np.unique(a[a != 7], return_counts=True)
        non4 = vals != 4
        if np.any(non4):
            bg = int(vals[non4][np.argmax(counts[non4])])
        else:
            bg = 0
    else:
        bg = 0

    # If there is no object, just remove markers.
    if not np.any(a == 4):
        out = a.copy()
        out[out == 7] = bg
        return out

    # If there are no markers, nothing to move.
    if not np.any(a == 7):
        return a.copy()

    # Find 4-connected components of color 4.
    comp_id = np.full((H, W), -1, dtype=int)
    comps = []
    for r in range(H):
        for c in range(W):
            if a[r, c] == 4 and comp_id[r, c] == -1:
                cid = len(comps)
                cells = []
                stack = [(r, c)]
                comp_id[r, c] = cid

                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dr, dc in dirs4:
                        nx, ny = x + dr, y + dc
                        if (
                            0 <= nx < H
                            and 0 <= ny < W
                            and a[nx, ny] == 4
                            and comp_id[nx, ny] == -1
                        ):
                            comp_id[nx, ny] = cid
                            stack.append((nx, ny))

                comps.append(cells)

    if not comps:
        out = a.copy()
        out[out == 7] = bg
        return out

    # Assign each 7 marker to every adjacent 4-component.
    marker_lists = [[] for _ in comps]
    for r in range(H):
        for c in range(W):
            if a[r, c] == 7:
                touched = set()
                for dr, dc in dirs4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and a[nr, nc] == 4:
                        cid = comp_id[nr, nc]
                        if cid >= 0:
                            touched.add(cid)
                for cid in touched:
                    marker_lists[cid].append((r, c))

    # Build information for each marked component.
    infos = []
    for cid, markers in enumerate(marker_lists):
        if not markers:
            continue

        cells = comps[cid]
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        br1, br2 = min(rs), max(rs)
        bc1, bc2 = min(cs), max(cs)

        mrs = [r for r, _ in markers]
        mcs = [c for _, c in markers]
        mr1, mr2 = min(mrs), max(mrs)
        mc1, mc2 = min(mcs), max(mcs)

        # For each marker, record which directions lead to its component.
        dir_counts = {(1, 0): 0, (-1, 0): 0, (0, 1): 0, (0, -1): 0}
        marker_dir_map = {}
        for r, c in markers:
            present = set()
            for dr, dc in dirs4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < H
                    and 0 <= nc < W
                    and a[nr, nc] == 4
                    and comp_id[nr, nc] == cid
                ):
                    present.add((dr, dc))
            marker_dir_map[(r, c)] = present
            for d in present:
                dir_counts[d] += 1

        possible_row = set()
        possible_col = set()
        if dir_counts[(1, 0)] > 0:
            possible_row.add(1)
        if dir_counts[(-1, 0)] > 0:
            possible_row.add(-1)
        if dir_counts[(0, 1)] > 0:
            possible_col.add(1)
        if dir_counts[(0, -1)] > 0:
            possible_col.add(-1)

        # Fallback if a marker is not directly adjacent due to unusual input.
        if not possible_row and not possible_col:
            if mr1 < br1:
                possible_row.add(1)
            elif mr2 > br2:
                possible_row.add(-1)
            if mc1 < bc1:
                possible_col.add(1)
            elif mc2 > bc2:
                possible_col.add(-1)
        if not possible_row and not possible_col:
            possible_row.add(1)

        infos.append(
            {
                "id": cid,
                "cells": cells,
                "cell_count": len(cells),
                "markers": markers,
                "marker_dir_map": marker_dir_map,
                "mr1": mr1,
                "mr2": mr2,
                "mc1": mc1,
                "mc2": mc2,
                "mr_center": (mr1 + mr2) / 2.0,
                "mc_center": (mc1 + mc2) / 2.0,
                "possible_row": possible_row,
                "possible_col": possible_col,
            }
        )

    if not infos:
        out = a.copy()
        out[out == 7] = bg
        return crop_to_4(out)

    # If only one marked object exists, just remove markers and crop.
    if len(infos) == 1:
        out = a.copy()
        out[out == 7] = bg
        return crop_to_4(out)

    # If there are extra marked objects, keep the two largest.
    if len(infos) > 2:
        infos = sorted(infos, key=lambda x: x["cell_count"], reverse=True)[:2]

    info1, info2 = infos

    def effective_bbox(info, axis, d):
        """
        Bounding box of the marker cells that actually correspond to direction d
        along the chosen joining axis. Falls back to all markers if needed.
        """
        if d is None or d == 0 or not info["markers"]:
            return info["mr1"], info["mr2"], info["mc1"], info["mc2"]

        if axis == "row":
            desired = (1, 0) if d == 1 else (-1, 0)
        else:
            desired = (0, 1) if d == 1 else (0, -1)

        selected = []
        for m, ds in info["marker_dir_map"].items():
            if desired in ds:
                selected.append(m)

        if not selected:
            for m, ds in info["marker_dir_map"].items():
                if axis == "row":
                    if any(dr != 0 for dr, _ in ds):
                        selected.append(m)
                else:
                    if any(dc != 0 for _, dc in ds):
                        selected.append(m)

        if not selected:
            selected = info["markers"]

        sel_rs = [r for r, _ in selected]
        sel_cs = [c for _, c in selected]
        return min(sel_rs), max(sel_rs), min(sel_cs), max(sel_cs)

    def choose_axis(i1, i2):
        """Choose whether the joining axis is vertical ('row') or horizontal ('col')."""
        score_row = 0
        score_col = 0

        if i1["possible_row"] and i2["possible_row"]:
            score_row += 20
            for d in i1["possible_row"]:
                if -d in i2["possible_row"]:
                    score_row += 100
                    break
        elif i1["possible_row"] or i2["possible_row"]:
            score_row += 5

        if i1["possible_col"] and i2["possible_col"]:
            score_col += 20
            for d in i1["possible_col"]:
                if -d in i2["possible_col"]:
                    score_col += 100
                    break
        elif i1["possible_col"] or i2["possible_col"]:
            score_col += 5

        if score_row == score_col:
            row_orient = 0
            col_orient = 0
            for inf in (i1, i2):
                if inf["mc2"] - inf["mc1"] >= inf["mr2"] - inf["mr1"]:
                    row_orient += 1
                else:
                    col_orient += 1

            if row_orient > col_orient:
                score_row += 10
            elif col_orient > row_orient:
                score_col += 10

        if score_row > score_col:
            return "row"
        if score_col > score_row:
            return "col"

        # Tie: use larger separation of marker centroids.
        r1 = (i1["mr1"] + i1["mr2"]) / 2.0
        r2 = (i2["mr1"] + i2["mr2"]) / 2.0
        c1 = (i1["mc1"] + i1["mc2"]) / 2.0
        c2 = (i2["mc1"] + i2["mc2"]) / 2.0
        if abs(r1 - r2) >= abs(c1 - c2):
            return "row"
        return "col"

    def axis_offset(src, tgt, axis, d_s):
        """Translation distance along the joining axis for a candidate direction."""
        src_bbox = effective_bbox(src, axis, d_s)
        tgt_bbox = effective_bbox(tgt, axis, -d_s)

        if axis == "row":
            if d_s == 1:
                return tgt_bbox[0] - src_bbox[1] - 1
            if d_s == -1:
                return tgt_bbox[1] + 1 - src_bbox[0]
            return 0
        else:
            if d_s == 1:
                return tgt_bbox[2] - src_bbox[3] - 1
            if d_s == -1:
                return tgt_bbox[3] + 1 - src_bbox[2]
            return 0

    axis = choose_axis(info1, info2)

    # Choose which marked object moves. Prefer the smaller one, but respect valid
    # opposing marker directions.
    best = None
    best_score = -1
    for src, tgt in ((info1, info2), (info2, info1)):
        src_set = src["possible_row"] if axis == "row" else src["possible_col"]
        tgt_set = tgt["possible_row"] if axis == "row" else tgt["possible_col"]

        score = 0
        d_s = 0
        best_off = None

        for d in sorted(src_set):
            if -d in tgt_set:
                off = axis_offset(src, tgt, axis, d)
                if (
                    best_off is None
                    or abs(off) < abs(best_off)
                    or (abs(off) == abs(best_off) and d == 1)
                ):
                    best_off = off
                    d_s = d

        if d_s != 0:
            score += 100

        if src["cell_count"] < tgt["cell_count"]:
            score += 20
        elif src["cell_count"] == tgt["cell_count"]:
            if axis == "row":
                if src["mr_center"] > tgt["mr_center"]:
                    score += 5
                elif src["mr_center"] == tgt["mr_center"] and src["mr2"] > tgt["mr2"]:
                    score += 2
            else:
                if src["mc_center"] > tgt["mc_center"]:
                    score += 5
                elif src["mc_center"] == tgt["mc_center"] and src["mc2"] > tgt["mc2"]:
                    score += 2

        if best is None or score > best_score:
            best_score = score
            best = (src, tgt, d_s)

    src, tgt, d_s = best
    d_t = -d_s if d_s != 0 else 0

    src_bbox = effective_bbox(src, axis, d_s)
    tgt_bbox = effective_bbox(tgt, axis, d_t)

    # Compute translation.
    if axis == "row":
        if d_s == 1:
            dr = tgt_bbox[0] - src_bbox[1] - 1
        elif d_s == -1:
            dr = tgt_bbox[1] + 1 - src_bbox[0]
        else:
            if src["mr_center"] > tgt["mr_center"]:
                dr = tgt_bbox[0] - src_bbox[1] - 1
            else:
                dr = tgt_bbox[1] + 1 - src_bbox[0]
        dc = tgt_bbox[2] - src_bbox[2]
    else:
        dr = tgt_bbox[0] - src_bbox[0]
        if d_s == 1:
            dc = tgt_bbox[2] - src_bbox[3] - 1
        elif d_s == -1:
            dc = tgt_bbox[3] + 1 - src_bbox[2]
        else:
            if src["mc_center"] > tgt["mc_center"]:
                dc = tgt_bbox[2] - src_bbox[3] - 1
            else:
                dc = tgt_bbox[3] + 1 - src_bbox[2]

    # Build output grid.
    base = a.copy()
    base[base == 7] = bg

    source_cells = src["cells"]
    if not source_cells:
        return crop_to_4(base)

    src_new_rmin = min(r + dr for r, _ in source_cells)
    src_new_rmax = max(r + dr for r, _ in source_cells)
    src_new_cmin = min(c + dc for _, c in source_cells)
    src_new_cmax = max(c + dc for _, c in source_cells)

    new_r_min = min(0, src_new_rmin)
    new_r_max = max(H - 1, src_new_rmax)
    new_c_min = min(0, src_new_cmin)
    new_c_max = max(W - 1, src_new_cmax)

    new_H = new_r_max - new_r_min + 1
    new_W = new_c_max - new_c_min + 1

    new_grid = np.full((new_H, new_W), bg, dtype=int)
    off_r = -new_r_min
    off_c = -new_c_min

    # Copy the original grid with markers removed.
    new_grid[off_r:off_r + H, off_c:off_c + W] = base

    # Remove the moving object from its original position.
    for r, c in source_cells:
        nr, nc = off_r + r, off_c + c
        if 0 <= nr < new_H and 0 <= nc < new_W:
            new_grid[nr, nc] = bg

    # Place the moving object at its translated position.
    for r, c in source_cells:
        nr, nc = off_r + r + dr, off_c + c + dc
        if 0 <= nr < new_H and 0 <= nc < new_W:
            new_grid[nr, nc] = 4

    return crop_to_4(new_grid)