import numpy as np
from collections import defaultdict
from itertools import product
from scipy import ndimage


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the ARC input grid.

    Rule summary
    ------------
    * 8 is the background.
    * There are 2-pixel-wide rectangular "blocks".
    * Each block has three ports along its long axis and emits straight lines
      in one of the two perpendicular directions.
    * Single-pixel non-background cells are color markers.  They are used to
      decide:
        - the emission direction for a block when they lie on one side of it,
        - the global lane colors for the three ports.
    * Vertical blocks normally use port order top-to-bottom as lanes 0,1,2.
      Horizontal blocks normally use left-to-right as lanes 2,1,0.
      A flipped block reverses this lane order.  Flip is inferred from the
      "1,1" cap when present, otherwise from a uniform "4,4" cap.
    * Lines are drawn from block edges to the grid border, painting only
      background cells so existing objects/markers are preserved.
    """
    if grid.size == 0:
        return grid.copy()

    out = np.array(grid, dtype=int).copy()
    H, W = out.shape

    # Background is the most common color (normally 8).
    vals, counts = np.unique(out, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    # Non-background mask and connected components (4-connectivity).
    mask = out != bg
    if not mask.any():
        return out

    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)
    labels, ncomp = ndimage.label(mask, structure=structure)

    markers = []
    blocks = []

    for cid in range(1, ncomp + 1):
        ys, xs = np.where(labels == cid)
        if len(ys) == 0:
            continue

        if len(ys) == 1:
            r, c = int(ys[0]), int(xs[0])
            markers.append({
                "id": len(markers),
                "color": int(out[r, c]),
                "r": r,
                "c": c,
            })
            continue

        rmin, rmax = int(ys.min()), int(ys.max())
        cmin, cmax = int(xs.min()), int(xs.max())
        h = rmax - rmin + 1
        w = cmax - cmin + 1

        # Blocks are 2 pixels wide/tall and at least moderately long.
        if min(h, w) == 2 and max(h, w) >= 4:
            vertical = h > w
            blocks.append({
                "id": len(blocks),
                "vertical": vertical,
                "rmin": rmin,
                "rmax": rmax,
                "cmin": cmin,
                "cmax": cmax,
                "L": h if vertical else w,
            })

    if not blocks or not markers:
        return out

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def active_direction(b):
        """
        Fallback direction from the block's internal colors.
        Prefer sides with more 'active' colors 1/2, then fewer 4s.
        """
        if b["vertical"]:
            cols = (b["cmin"], b["cmax"])
            sides = ("L", "R")
        else:
            rows = (b["rmin"], b["rmax"])
            sides = ("U", "D")

        best = None
        best_score = None

        for idx, side in enumerate(sides):
            active = 0
            non4 = 0
            if b["vertical"]:
                c = cols[idx]
                for r in range(b["rmin"], b["rmax"] + 1):
                    v = out[r, c]
                    if v == 1 or v == 2:
                        active += 1
                    if v != 4:
                        non4 += 1
            else:
                r = rows[idx]
                for c in range(b["cmin"], b["cmax"] + 1):
                    v = out[r, c]
                    if v == 1 or v == 2:
                        active += 1
                    if v != 4:
                        non4 += 1

            score = (active, non4)
            if best_score is None or score > best_score:
                best_score = score
                best = side

        return best

    def cap_priority(pair):
        """
        Priority for a uniform end cap.  Lower number is stronger.
        A (1,1) cap is strongest, then (4,4), then (3,3), then (2,2).
        """
        a, b = int(pair[0]), int(pair[1])
        if a == b == 1:
            return 0
        if a == b == 4:
            return 1
        if a == b == 3:
            return 2
        if a == b == 2:
            return 3
        return 10

    def compute_flip(b):
        """
        Infer whether the block's canonical lane order is reversed.

        For vertical blocks, lane 0 normally corresponds to the top end.
        For horizontal blocks, lane 0 normally corresponds to the right end.

        If a strong uniform cap exists, the end with that cap is treated as
        the lane-0 end.  In challenge-like blocks, the (1,1) end is the
        canonical lane-0 end, so a block with the (1,1) cap at the bottom is
        flipped vertically.
        """
        if b["vertical"]:
            top = (out[b["rmin"], b["cmin"]], out[b["rmin"], b["cmax"]])
            bottom = (out[b["rmax"], b["cmin"]], out[b["rmax"], b["cmax"]])
            top_p = cap_priority(top)
            bot_p = cap_priority(bottom)

            if top_p < bot_p:
                return False       # top is canonical lane-0 end
            if bot_p < top_p:
                return True        # bottom is canonical lane-0 end
            return False
        else:
            left = (out[b["rmin"], b["cmin"]], out[b["rmax"], b["cmin"]])
            right = (out[b["rmin"], b["cmax"]], out[b["rmax"], b["cmax"]])
            left_p = cap_priority(left)
            right_p = cap_priority(right)

            # Default horizontal lane-0 is the right end.
            if left_p < right_p:
                return True        # left is canonical lane-0 end
            return False

    def valid_offsets(tup, L):
        s = sorted({int(x) for x in tup if 0 <= int(x) < L})
        if len(s) == 3:
            return tuple(s)
        return None

    def candidate_offsets(b, marker_offsets):
        """
        Generate plausible port-offset sets for this block.
        Standard sets are preferred; marker-aligned sets are added as fallbacks.
        """
        L = b["L"]
        sets = {}

        def add(tup, score):
            t = valid_offsets(tup, L)
            if t is not None:
                sets[t] = max(sets.get(t, 0), score)

        if b["vertical"]:
            if L <= 6:
                add((1, 3, 4), 100)
                add((1, 4, 5), 90)
                add((0, 1, 4), 85)
                add((0, 1, 3), 75)
                add((1, 2, 4), 70)
                add((1, 3, 5), 80)
            elif L == 7:
                add((1, 4, 5), 100)
                add((1, 4, 6), 95)
                add((1, 3, 5), 90)
                add((0, 1, 4), 80)
                add((1, 2, 4), 70)
                add((1, 3, 6), 70)
            else:
                add((1, 3, 5), 100)
                add((1, 4, 5), 90)
                add((1, 4, 6), 85)
                add((1, 3, 6), 85)
                add((0, 1, 4), 75)
        else:
            if L <= 6:
                add((1, 3, 4), 100)
                add((1, 3, 5), 90)
                add((0, 1, 4), 85)
                add((1, 2, 4), 70)
            elif L == 7:
                add((1, 3, 5), 100)
                add((1, 4, 5), 90)
                add((3, 5, 7), 85)
                add((0, 1, 4), 80)
                add((1, 2, 4), 70)
            elif L == 9:
                add((3, 5, 7), 100)
                add((1, 3, 5), 90)
                add((1, 4, 5), 85)
                add((1, 3, 6), 80)
                add((0, 2, 4), 75)
            else:
                add((3, 5, 7), 100)
                add((1, 3, 5), 90)
                add((1, 4, 5), 85)

        # Add a few sets that explicitly include marker-aligned offsets.
        for _, _, o in marker_offsets:
            o = int(o)
            add((o, o + 1, o + 2), 65)
            add((o - 1, o, o + 1), 65)
            add((o, o + 2, o + 4), 60)
            add((0, o, L - 1), 65)
            add((1, o, L - 1), 60)
            add((o, L - 2, L - 1), 60)

        # Keep the highest-scoring candidates only.
        ordered = sorted(sets.items(), key=lambda x: (-x[1], x[0]))
        return ordered[:6]

    def lane_for_index(b, idx):
        """Map a sorted port index to a global lane index."""
        if b["vertical"]:
            return idx if not b["flip"] else 2 - idx
        else:
            return 2 - idx if not b["flip"] else idx

    def score_combo(combo):
        """
        Score a full choice of one option per block.
        Returns (score, lane_color).
        """
        lane_values = {}
        conflict_lanes = set()
        used_markers = set()

        base = 0
        active_match = 0
        marker_match = 0

        for b, opt in zip(blocks, combo):
            base += opt["base"]
            if opt["dir"] == b["active_dir"]:
                active_match += 1
            marker_match += len(opt["assignments"])

            for lane, color, mid in opt["assignments"]:
                used_markers.add(mid)
                if lane not in lane_values:
                    lane_values[lane] = color
                elif lane_values[lane] != color:
                    conflict_lanes.add(lane)

        distinct = sum(1 for l in lane_values if l not in conflict_lanes)

        # Consistency bonus: same block type/flip/offsets should match.
        groups = defaultdict(int)
        for b, opt in zip(blocks, combo):
            key = (b["vertical"], b["L"], b["flip"], opt["offsets"])
            groups[key] += 1
        consistency = 0
        for cnt in groups.values():
            if cnt > 1:
                consistency += 5 * cnt * (cnt - 1) // 2

        total = base
        total += 10 * len(used_markers)
        total += 20 * distinct
        total -= 50 * len(conflict_lanes)
        total += 5 * active_match
        total += 10 * marker_match

        if len(used_markers) == len(markers):
            total += 50
        if distinct == 3:
            total += 30

        total += consistency
        return total, lane_values

    # ------------------------------------------------------------------
    # Build block options
    # ------------------------------------------------------------------
    for b in blocks:
        b["active_dir"] = active_direction(b)
        b["flip"] = compute_flip(b)

        # Find markers that share the block's long-axis coordinate.
        b["marker_offsets"] = []
        for m in markers:
            if b["vertical"]:
                if b["rmin"] <= m["r"] <= b["rmax"] and (m["c"] < b["cmin"] or m["c"] > b["cmax"]):
                    side = "L" if m["c"] < b["cmin"] else "R"
                    b["marker_offsets"].append((m["id"], side, m["r"] - b["rmin"]))
            else:
                if b["cmin"] <= m["c"] <= b["cmax"] and (m["r"] < b["rmin"] or m["r"] > b["rmax"]):
                    side = "U" if m["r"] < b["rmin"] else "D"
                    b["marker_offsets"].append((m["id"], side, m["c"] - b["cmin"]))

        cand = candidate_offsets(b, b["marker_offsets"])
        opts = []
        possible_sides = ("L", "R") if b["vertical"] else ("U", "D")

        for offsets, score in cand:
            idx_map = {o: i for i, o in enumerate(offsets)}
            aligned_sides = set()
            for _, side, o in b["marker_offsets"]:
                if o in idx_map:
                    aligned_sides.add(side)

            dirs = list(aligned_sides)
            if b["active_dir"] not in dirs:
                dirs.append(b["active_dir"])

            for d in dirs:
                assignments = []
                for mid, side, o in b["marker_offsets"]:
                    if side == d and o in idx_map:
                        lane = lane_for_index(b, idx_map[o])
                        if 0 <= lane <= 2:
                            assignments.append((lane, markers[mid]["color"], mid))

                opts.append({
                    "offsets": offsets,
                    "dir": d,
                    "base": score + (5 if d == b["active_dir"] else 0),
                    "assignments": assignments,
                })

        # Remove duplicate (offsets, dir) options, keeping the better base score.
        best = {}
        for o in opts:
            key = (o["offsets"], o["dir"])
            if key not in best or o["base"] > best[key]["base"]:
                best[key] = o
        b["options"] = list(best.values())

        # Safety fallback.
        if not b["options"]:
            if b["vertical"]:
                offsets = tuple([1, 3, 5])
                if offsets[2] >= b["L"]:
                    offsets = tuple([0, 1, 2])
            else:
                offsets = tuple([1, 3, 5])
                if offsets[2] >= b["L"]:
                    offsets = tuple([0, 1, 2])
            b["options"] = [{
                "offsets": offsets,
                "dir": b["active_dir"],
                "base": 0,
                "assignments": [],
            }]

    # ------------------------------------------------------------------
    # Choose the best combination of block options
    # ------------------------------------------------------------------
    if len(blocks) <= 5:
        best_score = -10**18
        best_combo = None
        best_lane_color = {}

        for combo in product(*[b["options"] for b in blocks]):
            sc, lc = score_combo(combo)
            if sc > best_score:
                best_score = sc
                best_combo = combo
                best_lane_color = lc
    else:
        # Greedy fallback for larger instances.
        best_combo = [None] * len(blocks)
        best_lane_color = {}

        for i in range(len(blocks)):
            best_opt = None
            best_sc = -10**18
            best_lc = {}

            for opt in blocks[i]["options"]:
                tmp_combo = best_combo.copy()
                tmp_combo[i] = opt
                # Add tentative choices for later blocks to score whole combo.
                for j in range(i + 1, len(blocks)):
                    tmp_combo[j] = blocks[j]["options"][0]

                sc, lc = score_combo(tmp_combo)
                if sc > best_sc:
                    best_sc = sc
                    best_opt = opt
                    best_lc = lc

            best_combo[i] = best_opt

        sc, best_lane_color = score_combo(best_combo)

    # ------------------------------------------------------------------
    # Fill any missing global lane colors
    # ------------------------------------------------------------------
    used_colors = set(best_lane_color.values())
    remaining = [m for m in markers if m["color"] not in used_colors]

    for lane in (0, 1, 2):
        if lane not in best_lane_color:
            if remaining:
                # Simple deterministic fallback: smallest color first.
                remaining.sort(key=lambda m: m["color"])
                best_lane_color[lane] = remaining[0]["color"]
                remaining.pop(0)
            else:
                best_lane_color[lane] = bg

    # ------------------------------------------------------------------
    # Draw the output lines
    # ------------------------------------------------------------------
    def paint_vertical_line(r, c_start, direction, color):
        if direction == "L":
            for c in range(c_start - 1, -1, -1):
                if out[r, c] == bg:
                    out[r, c] = color
        else:  # R
            for c in range(c_start + 1, W):
                if out[r, c] == bg:
                    out[r, c] = color

    def paint_horizontal_line(c, r_start, direction, color):
        if direction == "U":
            for r in range(r_start - 1, -1, -1):
                if out[r, c] == bg:
                    out[r, c] = color
        else:  # D
            for r in range(r_start + 1, H):
                if out[r, c] == bg:
                    out[r, c] = color

    for b, opt in zip(blocks, best_combo):
        offsets = opt["offsets"]
        direction = opt["dir"]

        for i, o in enumerate(offsets):
            lane = lane_for_index(b, i)
            color = int(best_lane_color.get(lane, bg))
            if color == bg:
                continue

            if b["vertical"]:
                r = b["rmin"] + o
                c_start = b["cmin"] if direction == "L" else b["cmax"]
                paint_vertical_line(r, c_start, direction, color)
            else:
                c = b["cmin"] + o
                r_start = b["rmin"] if direction == "U" else b["rmax"]
                paint_horizontal_line(c, r_start, direction, color)

    return out