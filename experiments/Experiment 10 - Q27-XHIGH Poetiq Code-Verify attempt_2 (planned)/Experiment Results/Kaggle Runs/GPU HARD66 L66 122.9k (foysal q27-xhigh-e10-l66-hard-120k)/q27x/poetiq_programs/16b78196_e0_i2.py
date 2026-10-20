import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Transform the grid by treating the largest connected component as a
    horizontal/vertical "band".  The other components are pieces.

    The rule implemented here is:
      * detect notches/slots in the boundary of the band;
      * build small interlocking chains of pieces whose parallel dimension
        matches the slot size;
      * place the best chain into each suitable slot, allowing pieces to
        be shifted and rotated/flipped;
      * draw the band first, then draw the pieces, letting pieces cover
        band cells only in the slot opening, while preserving side walls.
    """
    grid = np.array(grid, dtype=int)
    if grid.size == 0:
        return grid.copy()

    gh, gw = grid.shape

    # ------------------------------------------------------------
    # Connected components (4-connected)
    # ------------------------------------------------------------
    def get_components():
        visited = np.zeros((gh, gw), dtype=bool)
        comps = []

        for r0 in range(gh):
            for c0 in range(gw):
                if grid[r0, c0] == 0 or visited[r0, c0]:
                    continue

                color = int(grid[r0, c0])
                stack = [(r0, c0)]
                visited[r0, c0] = True
                pts = []

                while stack:
                    r, c = stack.pop()
                    pts.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < gh and 0 <= nc < gw:
                            if not visited[nr, nc] and int(grid[nr, nc]) == color:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                rmin = min(r for r, _ in pts)
                cmin = min(c for _, c in pts)
                rmax = max(r for r, _ in pts)
                cmax = max(c for _, c in pts)

                mask = [[0] * (cmax - cmin + 1) for _ in range(rmax - rmin + 1)]
                for r, c in pts:
                    mask[r - rmin][c - cmin] = 1

                area = len(pts)
                comps.append({
                    "color": color,
                    "pts": pts,
                    "mask": mask,
                    "h": rmax - rmin + 1,
                    "w": cmax - cmin + 1,
                    "area": area,
                    "center_r": sum(r for r, _ in pts) / float(area),
                    "center_c": sum(c for _, c in pts) / float(area),
                })
        return comps

    comps = get_components()
    if not comps:
        return np.zeros_like(grid)

    # Largest component is the band.
    band = max(comps, key=lambda x: x["area"])
    band_color = band["color"]

    band_mask = np.zeros((gh, gw), dtype=bool)
    for r, c in band["pts"]:
        band_mask[r, c] = True

    # Other components are pieces.
    pieces = [c for c in comps if c is not band]
    for i, p in enumerate(pieces):
        p["id"] = i

    if not pieces:
        out = np.zeros_like(grid)
        out[band_mask] = band_color
        return out

    band_orient = "h" if band["w"] >= band["h"] else "v"

    # ------------------------------------------------------------
    # Mask utilities
    # ------------------------------------------------------------
    def rot90(m):
        return [list(row) for row in zip(*m[::-1])]

    def flip_h(m):
        return [row[::-1] for row in m]

    def unique_mats(mats):
        seen = set()
        res = []
        for m in mats:
            t = tuple(tuple(row) for row in m)
            if t not in seen:
                seen.add(t)
                res.append(m)
        return res

    def orientations(m):
        mats = []
        cur = [row[:] for row in m]
        for _ in range(4):
            mats.append(cur)
            mats.append(flip_h(cur))
            cur = rot90(cur)
        return unique_mats(mats)

    # Precompute orientations for all pieces.
    cid_counter = [0]
    for p in pieces:
        p["orients"] = orientations(p["mask"])

    # ------------------------------------------------------------
    # Slot detection in the band boundary.
    # ------------------------------------------------------------
    def find_slots():
        slots = []

        if band_orient == "h":
            topmost = [-1] * gw
            bottommost = [-1] * gw

            for c in range(gw):
                idx = np.where(band_mask[:, c])[0]
                if len(idx):
                    topmost[c] = int(idx[0])
                    bottommost[c] = int(idx[-1])

            # Top slots.
            valid_t = [x for x in topmost if x >= 0]
            if valid_t:
                base = min(valid_t)
                c = 0
                while c < gw:
                    if topmost[c] > base and topmost[c] >= 0:
                        s = c
                        while c < gw and topmost[c] > base and topmost[c] >= 0:
                            c += 1
                        e = c - 1
                        if s > 0 and e < gw - 1:
                            if topmost[s - 1] == base and topmost[e + 1] == base:
                                L = e - s + 1
                                slots.append({
                                    "side": "top",
                                    "orient": "h",
                                    "baseline": base,
                                    "p0": s - 1,
                                    "L": L,
                                    "H": L + 2,
                                })
                    else:
                        c += 1

            # Bottom slots.
            valid_b = [x for x in bottommost if x >= 0]
            if valid_b:
                base = max(valid_b)
                c = 0
                while c < gw:
                    if bottommost[c] < base and bottommost[c] >= 0:
                        s = c
                        while c < gw and bottommost[c] < base and bottommost[c] >= 0:
                            c += 1
                        e = c - 1
                        if s > 0 and e < gw - 1:
                            if bottommost[s - 1] == base and bottommost[e + 1] == base:
                                L = e - s + 1
                                slots.append({
                                    "side": "bottom",
                                    "orient": "h",
                                    "baseline": base,
                                    "p0": s - 1,
                                    "L": L,
                                    "H": L + 2,
                                })
                    else:
                        c += 1

        else:
            leftmost = [-1] * gh
            rightmost = [-1] * gh

            for r in range(gh):
                idx = np.where(band_mask[r, :])[0]
                if len(idx):
                    leftmost[r] = int(idx[0])
                    rightmost[r] = int(idx[-1])

            # Left slots.
            valid_l = [x for x in leftmost if x >= 0]
            if valid_l:
                base = min(valid_l)
                r = 0
                while r < gh:
                    if leftmost[r] > base and leftmost[r] >= 0:
                        s = r
                        while r < gh and leftmost[r] > base and leftmost[r] >= 0:
                            r += 1
                        e = r - 1
                        if s > 0 and e < gh - 1:
                            if leftmost[s - 1] == base and leftmost[e + 1] == base:
                                L = e - s + 1
                                slots.append({
                                    "side": "left",
                                    "orient": "v",
                                    "baseline": base,
                                    "p0": s - 1,
                                    "L": L,
                                    "H": L + 2,
                                })
                    else:
                        r += 1

            # Right slots.
            valid_r = [x for x in rightmost if x >= 0]
            if valid_r:
                base = max(valid_r)
                r = 0
                while r < gh:
                    if rightmost[r] < base and rightmost[r] >= 0:
                        s = r
                        while r < gh and rightmost[r] < base and rightmost[r] >= 0:
                            r += 1
                        e = r - 1
                        if s > 0 and e < gh - 1:
                            if rightmost[s - 1] == base and rightmost[e + 1] == base:
                                L = e - s + 1
                                slots.append({
                                    "side": "right",
                                    "orient": "v",
                                    "baseline": base,
                                    "p0": s - 1,
                                    "L": L,
                                    "H": L + 2,
                                })
                    else:
                        r += 1

        # Remove duplicates.
        out = []
        seen = set()
        for s in slots:
            key = (s["side"], s["baseline"], s["p0"], s["H"])
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    slots = find_slots()

    if not slots:
        # Fallback: keep the band and leave other pieces in place.
        out = grid.copy()
        return out

    # ------------------------------------------------------------
    # Candidate piece placements for a given slot parallel dimension.
    # ------------------------------------------------------------
    def make_candidates(slot_H, axis, pids):
        """
        axis = 'h' for horizontal band slots (parallel dimension is width).
        axis = 'v' for vertical band slots (parallel dimension is height).
        """
        raw = []
        for p in pieces:
            if p["id"] not in pids:
                continue

            for m in p["orients"]:
                h = len(m)
                w = len(m[0])

                if axis == "h":
                    dim = w
                    if dim == slot_H:
                        offs = [0]
                    elif dim == slot_H + 1:
                        offs = [0, 1]
                    else:
                        continue

                    for co in offs:
                        area = 0
                        sr = sc = 0
                        for i, row in enumerate(m):
                            for j, v in enumerate(row):
                                if v:
                                    area += 1
                                    sr += i
                                    sc += j
                        if area == 0:
                            continue
                        raw.append({
                            "pid": p["id"],
                            "color": p["color"],
                            "mask": m,
                            "h": h,
                            "w": w,
                            "core_off": co,
                            "center_r": sr / float(area),
                            "center_c": sc / float(area),
                            "cid": cid_counter[0],
                        })
                        cid_counter[0] += 1

                else:
                    dim = h
                    if dim == slot_H:
                        offs = [0]
                    elif dim == slot_H + 1:
                        offs = [0, 1]
                    else:
                        continue

                    for co in offs:
                        area = 0
                        sr = sc = 0
                        for i, row in enumerate(m):
                            for j, v in enumerate(row):
                                if v:
                                    area += 1
                                    sr += i
                                    sc += j
                        if area == 0:
                            continue
                        raw.append({
                            "pid": p["id"],
                            "color": p["color"],
                            "mask": m,
                            "h": h,
                            "w": w,
                            "core_off": co,
                            "center_r": sr / float(area),
                            "center_c": sc / float(area),
                            "cid": cid_counter[0],
                        })
                        cid_counter[0] += 1

        # Unique candidates.
        seen = set()
        res = []
        for c in raw:
            key = (c["pid"], c["core_off"], tuple(tuple(row) for row in c["mask"]))
            if key in seen:
                continue
            seen.add(key)
            c["uid"] = len(res)
            res.append(c)
        return res

    # ------------------------------------------------------------
    # Compatibility of two candidates in a chain.
    # ------------------------------------------------------------
    compat_cache = {}

    def get_compat(A, B, side, slot_H, axis):
        """
        A is the current inner/outermost piece, B is the next piece outward.
        Returns a normal offset d/dx, or None if they cannot interlock.
        """
        key = (A["cid"], B["cid"], side, slot_H, axis)
        if key in compat_cache:
            return compat_cache[key]

        best = None

        if axis == "h":
            # Vertical stacking (top/bottom band slots).
            Acore = [row[A["core_off"]:A["core_off"] + slot_H] for row in A["mask"]]
            Bcore = [row[B["core_off"]:B["core_off"] + slot_H] for row in B["mask"]]
            ha = len(Acore)
            hb = len(Bcore)

            for d in range(-hb, ha + 1):
                if side == "top" and d >= 0:
                    continue
                if side == "bottom" and d <= 0:
                    continue

                ov0 = max(0, d)
                ov1 = min(ha, d + hb)
                if ov0 >= ov1:
                    continue

                ok = True
                cover_rows = 0
                conflict = 0

                for i in range(ov0, ov1):
                    j = i - d
                    rowA = Acore[i]
                    rowB = Bcore[j]
                    row_cover = 0
                    for x in range(slot_H):
                        a = rowA[x]
                        b = rowB[x]
                        if a and b:
                            conflict += 1
                        if a or b:
                            row_cover += 1
                    if row_cover < slot_H:
                        ok = False
                        break
                    cover_rows += 1

                if ok:
                    val = (cover_rows, -conflict, -abs(d))
                    if best is None or val > best[0]:
                        best = (val, d)

        else:
            # Horizontal stacking (left/right band slots).
            Acore = A["mask"][A["core_off"]:A["core_off"] + slot_H]
            Bcore = B["mask"][B["core_off"]:B["core_off"] + slot_H]
            ha = slot_H
            wa = len(Acore[0])
            wb = len(Bcore[0])

            for dx in range(-wb, wa + 1):
                if side == "left" and dx >= 0:
                    continue
                if side == "right" and dx <= 0:
                    continue

                ov0 = max(0, dx)
                ov1 = min(wa, dx + wb)
                if ov0 >= ov1:
                    continue

                ok = True
                cover_cols = 0
                conflict = 0

                for a in range(ov0, ov1):
                    b = a - dx
                    col_cover = 0
                    for r in range(ha):
                        av = Acore[r][a]
                        bv = Bcore[r][b]
                        if av and bv:
                            conflict += 1
                        if av or bv:
                            col_cover += 1
                    if col_cover < ha:
                        ok = False
                        break
                    cover_cols += 1

                if ok:
                    val = (cover_cols, -conflict, -abs(dx))
                    if best is None or val > best[0]:
                        best = (val, dx)

        res = best[1] if best is not None else None
        compat_cache[key] = res
        return res

    # ------------------------------------------------------------
    # Generate chains for a slot type.
    # ------------------------------------------------------------
    chain_cache = {}
    all_pids = {p["id"] for p in pieces}

    def generate_chains(slot_H, axis, side):
        key = (slot_H, axis, side)
        if key in chain_cache:
            return chain_cache[key]

        cands = make_candidates(slot_H, axis, all_pids)
        chains = []
        seen = set()
        max_len = 4

        def dfs(items, used, outer, local, ext_min, ext_max):
            if len(items) >= 2:
                sig = tuple((c["uid"], l) for c, l in items)
                if sig not in seen:
                    seen.add(sig)
                    chains.append({
                        "items": list(items),
                        "pids": set(used),
                        "H": slot_H,
                        "axis": axis,
                        "side": side,
                    })
                    if len(chains) > 2000:
                        return True

            if len(items) >= max_len or len(chains) > 2000:
                return True

            for cand in cands:
                if cand["pid"] in used:
                    continue

                d = get_compat(outer, cand, side, slot_H, axis)
                if d is None:
                    continue

                new_local = local + d

                if axis == "h":
                    nmin = new_local
                    nmax = new_local + cand["h"]
                    if side == "top" and not (nmin < ext_min):
                        continue
                    if side == "bottom" and not (nmax > ext_max):
                        continue
                else:
                    nmin = new_local
                    nmax = new_local + cand["w"]
                    if side == "left" and not (nmin < ext_min):
                        continue
                    if side == "right" and not (nmax > ext_max):
                        continue

                if dfs(
                    items + [(cand, new_local)],
                    used | {cand["pid"]},
                    cand,
                    new_local,
                    min(ext_min, nmin),
                    max(ext_max, nmax),
                ):
                    return True

            return False

        for cand in cands:
            if axis == "h":
                ext_min = 0
                ext_max = cand["h"]
            else:
                ext_min = 0
                ext_max = cand["w"]

            dfs([(cand, 0)], {cand["pid"]}, cand, 0, ext_min, ext_max)

        chain_cache[key] = chains
        return chains

    # ------------------------------------------------------------
    # Score a chain placement in a slot.
    # ------------------------------------------------------------
    piece_center = {p["id"]: (p["center_r"], p["center_c"]) for p in pieces}

    def evaluate_chain(chain, slot):
        slot_H = slot["H"]
        axis = slot["orient"]
        side = slot["side"]
        p0 = slot["p0"]
        boundary = slot["baseline"]

        anchor = chain["items"][0][0]
        best = None

        if axis == "h":
            k_range = range(anchor["h"])
        else:
            k_range = range(anchor["w"])

        for k in k_range:
            penalty = 0.0
            good = 0.0
            dist = 0.0

            # Boundary-line match.
            if axis == "h":
                for x in range(slot_H):
                    cc = x + anchor["core_off"]
                    cell = anchor["mask"][k][cc] if 0 <= cc < anchor["w"] else 0
                    desired = 1 if 0 < x < slot_H - 1 else 0
                    gc = p0 + x

                    if 0 <= gc < gw:
                        if desired == 1 and cell == 0:
                            penalty += 5
                        if desired == 0 and cell == 1:
                            if band_mask[boundary, gc] and band_color > anchor["color"]:
                                penalty += 1
                            else:
                                penalty += 10
                        if desired == 1 and cell == 1:
                            good += 1

            else:
                for y in range(slot_H):
                    rr = y + anchor["core_off"]
                    cell = anchor["mask"][rr][k] if 0 <= rr < anchor["h"] else 0
                    desired = 1 if 0 < y < slot_H - 1 else 0
                    gr = p0 + y

                    if 0 <= gr < gh:
                        if desired == 1 and cell == 0:
                            penalty += 5
                        if desired == 0 and cell == 1:
                            if band_mask[gr, boundary] and band_color > anchor["color"]:
                                penalty += 1
                            else:
                                penalty += 10
                        if desired == 1 and cell == 1:
                            good += 1

            # Cell-level score.
            if axis == "h":
                anchor_full_top = boundary - k

                for cand, local in chain["items"]:
                    full_left = p0 - cand["core_off"]
                    full_top = anchor_full_top + local

                    for r, row in enumerate(cand["mask"]):
                        for c, v in enumerate(row):
                            if not v:
                                continue

                            R = full_top + r
                            C = full_left + c

                            if R < 0 or R >= gh or C < 0 or C >= gw:
                                penalty += 10
                                continue

                            p = c - cand["core_off"]

                            if band_mask[R, C]:
                                if 0 < p < slot_H - 1:
                                    # opening column/row: allowed
                                    pass
                                else:
                                    if band_color > cand["color"]:
                                        penalty += 1
                                    else:
                                        penalty += 15
                            else:
                                if p < 0 or p >= slot_H:
                                    penalty += 2

                    orig_r, orig_c = piece_center[cand["pid"]]
                    fr = full_top + cand["center_r"]
                    fc = full_left + cand["center_c"]
                    dist += abs(fr - orig_r) + abs(fc - orig_c)

            else:
                anchor_full_left = boundary - k

                for cand, local in chain["items"]:
                    full_top = p0 - cand["core_off"]
                    full_left = anchor_full_left + local

                    for r, row in enumerate(cand["mask"]):
                        for c, v in enumerate(row):
                            if not v:
                                continue

                            R = full_top + r
                            C = full_left + c

                            if R < 0 or R >= gh or C < 0 or C >= gw:
                                penalty += 10
                                continue

                            p = r - cand["core_off"]

                            if band_mask[R, C]:
                                if 0 < p < slot_H - 1:
                                    # opening row: allowed
                                    pass
                                else:
                                    if band_color > cand["color"]:
                                        penalty += 1
                                    else:
                                        penalty += 15
                            else:
                                if p < 0 or p >= slot_H:
                                    penalty += 2

                    orig_r, orig_c = piece_center[cand["pid"]]
                    fr = full_top + cand["center_r"]
                    fc = full_left + cand["center_c"]
                    dist += abs(fr - orig_r) + abs(fc - orig_c)

            score = 2.0 * good - penalty - 0.10 * dist
            if best is None or score > best[0]:
                best = (score, k)

        return best

    # ------------------------------------------------------------
    # Find best solo placement in a slot.
    # ------------------------------------------------------------
    def find_best_solo(slot, pids):
        slot_H = slot["H"]
        axis = slot["orient"]
        side = slot["side"]

        cands = make_candidates(slot_H, axis, set(pids))
        best = None

        for cand in cands:
            chain = {
                "items": [(cand, 0)],
                "pids": {cand["pid"]},
                "H": slot_H,
                "axis": axis,
                "side": side,
            }
            ev = evaluate_chain(chain, slot)
            if ev is None:
                continue
            score, k = ev
            if score < -12:
                continue
            if best is None or score > best[0]:
                best = (score, chain, k)

        return best

    # ------------------------------------------------------------
    # Greedy assignment of chains to slots.
    # ------------------------------------------------------------
    # Prefer larger slots first (they are usually more specific).
    slots.sort(key=lambda s: (-s["H"], s["p0"], s["baseline"], s["side"]))

    used = set()
    assignments = []

    for slot in slots:
        if used == all_pids:
            break

        unused = [p for p in pieces if p["id"] not in used]
        if not unused:
            break

        unused_ids = {p["id"] for p in unused}
        chains = generate_chains(slot["H"], slot["orient"], slot["side"])

        best = None
        for chain in chains:
            if not chain["pids"].issubset(unused_ids):
                continue
            ev = evaluate_chain(chain, slot)
            if ev is None:
                continue
            score, k = ev
            if score < -12:
                continue
            score += 3.0 * len(chain["items"])  # prefer interlocked chains
            if best is None or score > best[0]:
                best = (score, chain, k)

        # If no multi-chain, use the best solo piece for this slot.
        if best is None:
            solo = find_best_solo(slot, unused_ids)
            if solo is not None:
                score, chain, k = solo
                best = (score - 1.0, chain, k)

        if best is not None:
            _, chain, k = best
            assignments.append((slot, chain, k))
            used.update(chain["pids"])

    # ------------------------------------------------------------
    # Fallback for any remaining pieces: place them near the best slot.
    # ------------------------------------------------------------
    extra = []
    for p in pieces:
        if p["id"] in used:
            continue

        best = None
        for slot in slots:
            slot_H = slot["H"]
            axis = slot["orient"]

            for m in p["orients"]:
                h = len(m)
                w = len(m[0])

                if axis == "h":
                    lefts = range(slot["p0"] - 3, slot["p0"] + slot_H - w + 4)
                    tops = range(slot["baseline"] - h - 2, slot["baseline"] + 3)
                else:
                    lefts = range(slot["baseline"] - w - 2, slot["baseline"] + 3)
                    tops = range(slot["p0"] - 3, slot["p0"] + slot_H - h + 4)

                for top in tops:
                    for left in lefts:
                        score = 0.0
                        ok = True
                        for r, row in enumerate(m):
                            for c, v in enumerate(row):
                                if not v:
                                    continue
                                R = top + r
                                C = left + c
                                if R < 0 or R >= gh or C < 0 or C >= gw:
                                    ok = False
                                    break
                                if axis == "h":
                                    ppos = C - slot["p0"]
                                else:
                                    ppos = R - slot["p0"]

                                if 0 < ppos < slot_H - 1:
                                    score += 1
                                elif band_mask[R, C]:
                                    if band_color > p["color"]:
                                        score -= 2
                                    else:
                                        score -= 8
                                else:
                                    if ppos < 0 or ppos >= slot_H:
                                        score -= 0.5

                        if not ok:
                            continue

                        # proximity bonus
                        area = sum(sum(row) for row in m)
                        if area:
                            sr = sum(i * sum(row) for i, row in enumerate(m)) / float(area)
                            sc = sum(j * row[j] for row in m for j in range(len(row))) / float(area)
                            fr = top + sr
                            fc = left + sc
                            orig_r, orig_c = piece_center[p["id"]]
                            score -= 0.02 * (abs(fr - orig_r) + abs(fc - orig_c))

                        if best is None or score > best[0]:
                            best = (score, m, top, left, slot)

        if best is not None:
            _, m, top, left, slot = best
            extra.append((p["color"], m, top, left, slot))
        else:
            # Last resort: leave in original position.
            top = min(r for r, _ in p["pts"])
            left = min(c for _, c in p["pts"])
            extra.append((p["color"], p["mask"], top, left, None))

    # ------------------------------------------------------------
    # Draw output.
    # ------------------------------------------------------------
    out = np.zeros((gh, gw), dtype=int)
    out[band_mask] = band_color

    def draw_piece(color, mask, top, left, slot):
        if slot is None:
            slot_H = None
            p0 = None
            axis = None
        else:
            slot_H = slot["H"]
            p0 = slot["p0"]
            axis = slot["orient"]

        for r, row in enumerate(mask):
            for c, v in enumerate(row):
                if not v:
                    continue

                R = top + r
                C = left + c
                if R < 0 or R >= gh or C < 0 or C >= gw:
                    continue

                # Determine if this cell lies in the slot opening.
                opening = False
                if slot is not None:
                    if axis == "h":
                        p = C - p0
                    else:
                        p = R - p0
                    opening = (0 < p < slot_H - 1)

                cur = out[R, C]

                if cur == 0:
                    out[R, C] = color
                elif cur == band_color:
                    # Let pieces cover band only in the opening.
                    if opening:
                        out[R, C] = color
                    elif color > band_color and opening:
                        out[R, C] = color
                    # otherwise keep band
                else:
                    # Piece-piece overlap: higher color wins.
                    if color > cur:
                        out[R, C] = color

    # Draw chain assignments.
    for slot, chain, k in assignments:
        axis = slot["orient"]
        p0 = slot["p0"]
        boundary = slot["baseline"]

        if axis == "h":
            anchor_full_top = boundary - k
            for cand, local in chain["items"]:
                full_left = p0 - cand["core_off"]
                full_top = anchor_full_top + local
                draw_piece(cand["color"], cand["mask"], full_top, full_left, slot)
        else:
            anchor_full_left = boundary - k
            for cand, local in chain["items"]:
                full_top = p0 - cand["core_off"]
                full_left = anchor_full_left + local
                draw_piece(cand["color"], cand["mask"], full_top, full_left, slot)

    # Draw fallback extras.
    for color, mask, top, left, slot in extra:
        draw_piece(color, mask, top, left, slot)

    return out