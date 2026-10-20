import numpy as np
from collections import deque, defaultdict


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move 8/9 objects to 9-only anchor clusters.

    The input contains:
      - background color,
      - large fixed barriers/frame colors,
      - movable objects made of 8s and 9s plus colored endpoints,
      - anchor clusters made only of 9s.

    For every movable object, we try all 8 symmetries that place its 9s
    exactly onto some anchor 9-cluster. A candidate is scored by how well
    each endpoint color points toward the matching barrier color. Then a
    consistent assignment is selected and the objects are redrawn.
    """
    if grid is None:
        return grid

    g = np.array(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        return g

    h, w = g.shape

    # Background is the most frequent color.
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])

    def centroid(pts):
        if not pts:
            return (0.0, 0.0)
        return (
            float(sum(r for r, c in pts) / len(pts)),
            float(sum(c for r, c in pts) / len(pts)),
        )

    def label_mask(mask, eight=True):
        """Connected component labeling on a boolean mask."""
        labels = np.zeros((h, w), dtype=int)
        comps = []

        if eight:
            dirs = (
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1),
            )
        else:
            dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for i in range(h):
            for j in range(w):
                if mask[i, j] and labels[i, j] == 0:
                    cid = len(comps) + 1
                    q = deque([(i, j)])
                    labels[i, j] = cid
                    pix = []

                    while q:
                        r, c = q.popleft()
                        pix.append((r, c))

                        for dr, dc in dirs:
                            nr, nc = r + dr, c + dc
                            if (
                                0 <= nr < h and 0 <= nc < w
                                and mask[nr, nc]
                                and labels[nr, nc] == 0
                            ):
                                labels[nr, nc] = cid
                                q.append((nr, nc))

                    comps.append(pix)

        return comps

    # ------------------------------------------------------------------
    # Identify large fixed barriers.
    # Use 4-connectivity for barriers so that small diagonally touching
    # endpoint pixels are not accidentally merged into a barrier.
    # ------------------------------------------------------------------
    barrier_mask = np.zeros((h, w), dtype=bool)
    barrier_comps = defaultdict(list)
    threshold = max(10, int(h * w * 0.04))
    min_dim = min(h, w)

    for c in np.unique(g):
        c = int(c)
        if c == bg or c == 8 or c == 9:
            continue

        mask = (g == c)
        for pix in label_mask(mask, eight=False):
            if len(pix) < threshold:
                continue

            rs = [r for r, _ in pix]
            cs = [c_ for _, c_ in pix]
            height = max(rs) - min(rs)
            width = max(cs) - min(cs)

            # A true barrier/frame should be long in at least one direction.
            if max(height, width) < max(5, min_dim // 3):
                continue

            barrier_comps[c].append(pix)
            for r, c_ in pix:
                barrier_mask[r, c_] = True

    # ------------------------------------------------------------------
    # Label movable objects and anchors on non-background, non-barrier cells.
    # Use 8-connectivity so diagonal 9s and diagonal endpoint pixels are kept.
    # ------------------------------------------------------------------
    object_mask = (g != bg) & ~barrier_mask
    obj_comps = label_mask(object_mask, eight=True)

    sources = []
    targets = []
    source_mask = np.zeros((h, w), dtype=bool)

    for pix in obj_comps:
        colors = set(int(g[r, c]) for r, c in pix)
        has8 = 8 in colors
        has9 = 9 in colors

        if has8 and has9:
            src9 = []
            non9 = []
            endpoints = defaultdict(list)

            for r, c in pix:
                col = int(g[r, c])
                if col == 9:
                    src9.append((r, c))
                else:
                    non9.append((r, c, col))
                    if col != 8:
                        endpoints[col].append((r, c))

            if src9:
                sources.append({
                    "pixels": pix,
                    "9s": src9,
                    "non9": non9,
                    "endpoints": dict(endpoints),
                    "center": centroid(src9),
                })
                for r, c in pix:
                    source_mask[r, c] = True

        elif has9 and not has8:
            tgt9 = [(r, c) for r, c in pix if int(g[r, c]) == 9]
            if tgt9:
                targets.append({
                    "9s": tgt9,
                    "center": centroid(tgt9),
                    "count": len(tgt9),
                })

    if not sources or not targets:
        return g.copy()

    # ------------------------------------------------------------------
    # Barrier direction helpers.
    # ------------------------------------------------------------------
    def comp_info(pix):
        rs = [r for r, _ in pix]
        cs = [c for _, c in pix]
        minr, maxr = min(rs), max(rs)
        minc, maxc = min(cs), max(cs)
        return (
            minr, maxr, minc, maxc,
            maxr - minr, maxc - minc,
            len(pix),
            sum(rs) / len(pix),
            sum(cs) / len(pix),
            pix,
        )

    barrier_infos = defaultdict(list)
    for c, pixs in barrier_comps.items():
        for pix in pixs:
            barrier_infos[c].append(comp_info(pix))

    endpoint_colors = set()
    for src in sources:
        endpoint_colors.update(src["endpoints"].keys())

    # Fallback for colors with no large marked barrier: use smaller static
    # non-source components of that color.
    fallback_infos = {}
    for c in endpoint_colors:
        if barrier_infos.get(c):
            continue
        mask = (g == c) & ~source_mask & ~barrier_mask & (g != bg)
        if mask.any():
            infos = []
            for pix in label_mask(mask, eight=True):
                if pix:
                    infos.append(comp_info(pix))
            fallback_infos[c] = infos

    def comp_direction(info, center):
        """
        Return a cardinal direction from center to a barrier component,
        its distance, and component size.
        """
        minr, maxr, minc, maxc, height, width, size, meanr, meanc, pix = info
        cr, cc = center

        # Vertical line/barrier: decide left/right by columns.
        if height > width:
            if cc < minc:
                return (0, 1), float(minc - cc), size
            if cc > maxc:
                return (0, -1), float(cc - maxc), size

            medc = (minc + maxc) / 2.0
            if cc < medc:
                return (0, 1), abs(medc - cc), size
            return (0, -1), abs(medc - cc), size

        # Horizontal line/barrier: decide up/down by rows.
        if width > height:
            if cr < minr:
                return (1, 0), float(minr - cr), size
            if cr > maxr:
                return (-1, 0), float(cr - maxr), size

            medr = (minr + maxr) / 2.0
            if cr < medr:
                return (1, 0), abs(medr - cr), size
            return (-1, 0), abs(medr - cr), size

        # General blob/L-shape/full border: use nearest barrier pixel.
        best_dir = None
        best_d2 = float("inf")

        for r, c in pix:
            dr = float(r - cr)
            dc = float(c - cc)
            if abs(dr) < 1e-9 and abs(dc) < 1e-9:
                continue

            d2 = dr * dr + dc * dc
            if d2 < best_d2:
                if abs(dr) >= abs(dc):
                    dir_ = (-1, 0) if dr < 0 else (1, 0)
                else:
                    dir_ = (0, -1) if dc < 0 else (0, 1)
                best_dir = dir_
                best_d2 = d2

        if best_dir is None:
            return None, float("inf"), size
        return best_dir, best_d2 ** 0.5, size

    def direction_for_color(color, center):
        infos = barrier_infos.get(color)
        if not infos:
            infos = fallback_infos.get(color)
        if not infos:
            return None

        best_dir = None
        best_dist = float("inf")
        best_size = -1

        for info in infos:
            dir_, dist, size = comp_direction(info, center)
            if dir_ is None:
                continue

            if dist < best_dist - 1e-9 or (
                abs(dist - best_dist) < 1e-9 and size > best_size
            ):
                best_dir = dir_
                best_dist = dist
                best_size = size

        return best_dir

    # Desired endpoint direction for every target anchor.
    target_dirs = []
    for tgt in targets:
        dirs = {}
        for c in endpoint_colors:
            dirs[c] = direction_for_color(c, tgt["center"])
        target_dirs.append(dirs)

    # ------------------------------------------------------------------
    # Candidate generation.
    # ------------------------------------------------------------------
    blocked = (g != bg) & ~source_mask

    def apply_sym(v, sym):
        dr, dc = v
        if sym == 0:
            return dr, dc
        if sym == 1:
            return dc, -dr
        if sym == 2:
            return -dr, -dc
        if sym == 3:
            return -dc, dr
        if sym == 4:
            return dr, -dc
        if sym == 5:
            return -dr, dc
        if sym == 6:
            return dc, dr
        return -dc, -dr  # sym == 7

    def generate_candidates(si, ti):
        src = sources[si]
        tgt = targets[ti]

        if len(src["9s"]) != len(tgt["9s"]):
            return []

        src9 = src["9s"]
        tgt9 = tgt["9s"]
        tgt9set = set(tgt9)
        tr, tc = tgt["center"]

        dist = (
            (src["center"][0] - tr) ** 2 +
            (src["center"][1] - tc) ** 2
        ) ** 0.5

        cands = []
        seen = set()

        for sym in range(8):
            for ps in src9:
                for pt in tgt9:
                    transformed9 = []
                    ok = True

                    for p in src9:
                        dr = p[0] - ps[0]
                        dc = p[1] - ps[1]
                        ndr, ndc = apply_sym((dr, dc), sym)
                        q = (pt[0] + ndr, pt[1] + ndc)

                        if q not in tgt9set:
                            ok = False
                            break
                        transformed9.append(q)

                    if not ok or set(transformed9) != tgt9set:
                        continue

                    pix_map = {}
                    endpoints = defaultdict(list)
                    valid = True

                    for r, c, col in src["non9"]:
                        dr = r - ps[0]
                        dc = c - ps[1]
                        ndr, ndc = apply_sym((dr, dc), sym)
                        q = (pt[0] + ndr, pt[1] + ndc)

                        if q[0] < 0 or q[0] >= h or q[1] < 0 or q[1] >= w:
                            valid = False
                            break

                        # Fixed barriers and other anchors cannot be overwritten.
                        if blocked[q[0], q[1]]:
                            valid = False
                            break

                        if q in pix_map:
                            if pix_map[q] != col:
                                valid = False
                                break
                        else:
                            pix_map[q] = col
                            if col != 8:
                                endpoints[col].append(q)

                    if not valid:
                        continue

                    key = tuple(sorted(pix_map.items()))
                    if key in seen:
                        continue
                    seen.add(key)

                    score = 0.0
                    dirs = target_dirs[ti]

                    for color, pts in endpoints.items():
                        desired = dirs.get(color)
                        if desired is None:
                            continue

                        drd, dcd = desired
                        pos_sum = 0.0
                        neg_sum = 0.0
                        neg_count = 0

                        for r, c in pts:
                            proj = (r - tr) * drd + (c - tc) * dcd
                            if proj > 0:
                                pos_sum += proj
                            elif proj < 0:
                                neg_sum += -proj
                                neg_count += 1

                        if pts:
                            n = len(pts)
                            score += pos_sum / n
                            score -= 1.5 * (neg_sum / n)

                            if neg_count == 0:
                                score += 2.0
                            else:
                                score -= 2.0 * (neg_count / n)

                    # Small tie-breaker: prefer closer source-to-target moves.
                    score -= 0.001 * dist

                    cands.append({
                        "score": score,
                        "sym": sym,
                        "pix_map": pix_map,
                        "mask": frozenset(pix_map.keys()),
                    })

        cands.sort(key=lambda x: x["score"], reverse=True)
        if len(cands) > 20:
            cands = cands[:20]
        return cands

    ns = len(sources)
    nt = len(targets)

    cand_pairs = [[[] for _ in range(nt)] for _ in range(ns)]
    for i in range(ns):
        for j in range(nt):
            cand_pairs[i][j] = generate_candidates(i, j)

    if not any(cand_pairs[i][j] for i in range(ns) for j in range(nt)):
        return g.copy()

    # ------------------------------------------------------------------
    # Assignment.
    # ------------------------------------------------------------------
    assignment = []
    best_score = None
    best_assign = None

    # Exact backtracking for small instances.
    if ns <= nt and ns <= 6:
        if all(any(cand_pairs[i][j] for j in range(nt)) for i in range(ns)):
            order = sorted(
                range(ns),
                key=lambda i: sum(1 for j in range(nt) if cand_pairs[i][j]),
            )

            def bt(idx, used_targets, masks, cur, score):
                nonlocal best_score, best_assign

                if idx == len(order):
                    if best_score is None or score > best_score:
                        best_score = score
                        best_assign = cur[:]
                    return

                src_i = order[idx]
                options = []

                for j in range(nt):
                    if j in used_targets:
                        continue
                    for cand in cand_pairs[src_i][j]:
                        options.append((cand["score"], j, cand))

                options.sort(key=lambda x: x[0], reverse=True)

                # Prune extremely large branches, but keep enough options.
                limit = 50 if len(options) <= 50 else 25
                for _, j, cand in options[:limit]:
                    if any(cand["mask"] & m for m in masks):
                        continue

                    cur.append((src_i, j, cand))
                    masks.append(cand["mask"])

                    bt(idx + 1, used_targets | {j}, masks, cur, score + cand["score"])

                    masks.pop()
                    cur.pop()

            bt(0, set(), [], [], 0.0)

            if best_assign is not None:
                assignment = best_assign

    # Greedy fallback.
    if not assignment:
        all_cands = []
        for i in range(ns):
            for j in range(nt):
                for cand in cand_pairs[i][j]:
                    all_cands.append((cand["score"], i, j, cand))

        all_cands.sort(key=lambda x: x[0], reverse=True)

        used_sources = set()
        used_targets = set()
        masks = []

        for _, i, j, cand in all_cands:
            if i in used_sources or j in used_targets:
                continue
            if any(cand["mask"] & m for m in masks):
                continue

            used_sources.add(i)
            used_targets.add(j)
            masks.append(cand["mask"])
            assignment.append((i, j, cand))

    if not assignment:
        return g.copy()

    # ------------------------------------------------------------------
    # Render output.
    # ------------------------------------------------------------------
    out = g.copy()

    # Remove all movable source objects.
    out[source_mask] = bg

    # Draw assigned transformed objects.
    for _, _, cand in assignment:
        for (r, c), col in cand["pix_map"].items():
            if 0 <= r < h and 0 <= c < w:
                if out[r, c] == bg or out[r, c] == col:
                    out[r, c] = col

    # Preserve all anchor 9s.
    for tgt in targets:
        for r, c in tgt["9s"]:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = 9

    return out