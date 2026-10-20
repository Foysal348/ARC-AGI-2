import numpy as np
import sys

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble a framed puzzle.

    The input contains one "frame" region: a connected component of some color F
    whose bounding box contains only F and one "empty" color B.  The cells of B
    inside that bounding box are holes.  All other connected components are
    pieces.  The output is the bounding box of the frame, with the holes filled
    by the pieces without rotation or reflection.

    If there are more piece cells than hole cells, a subset of pieces is chosen.
    The subset is selected by a small dynamic program that favors constrained,
    non-zero, larger pieces.  The actual placement is found by exact-cover
    backtracking with MRV and forward checking.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    sys.setrecursionlimit(10000)

    def get_components(mask):
        """4-connected components as lists of (r, c)."""
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        comps = []

        for r0 in range(h):
            for c0 in range(w):
                if not mask[r0, c0] or seen[r0, c0]:
                    continue

                comp = []
                stack = [(r0, c0)]
                seen[r0, c0] = True

                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

                comps.append(comp)

        return comps

    # ------------------------------------------------------------------
    # Candidate frame selection.
    # ------------------------------------------------------------------
    candidates = []

    for F in np.unique(grid):
        F = int(F)
        mask_f = grid == F
        if not mask_f.any():
            continue

        for comp in get_components(mask_f):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r1, r2 = min(rs), max(rs)
            c1, c2 = min(cs), max(cs)

            sub = grid[r1:r2 + 1, c1:c2 + 1]
            uniq = np.unique(sub)

            # A valid frame region has only frame color F and empty color B.
            if len(uniq) != 2 or F not in uniq:
                continue

            B = int(uniq[0]) if int(uniq[0]) != F else int(uniq[1])
            if B == F:
                continue

            holes = int(np.sum(sub == B))
            if holes == 0:
                continue

            piece_count = int(np.sum(~((grid == F) | (grid == B))))
            if piece_count < holes:
                continue

            # Avoid candidates with an absurdly large excess, but keep
            # enough slack for distractor pieces.
            if piece_count > holes * 3 + 30:
                continue

            diff = piece_count - holes
            frame_area = len(comp)

            # Prefer the largest plausible puzzle, then smallest excess.
            candidates.append((
                -holes,
                diff,
                -frame_area,
                F, B, r1, r2, c1, c2
            ))

    candidates.sort()

    # ------------------------------------------------------------------
    # Exact-cover solver for one candidate.
    # ------------------------------------------------------------------
    def solve_candidate(F, B, r1, r2, c1, c2):
        sub = grid[r1:r2 + 1, c1:c2 + 1].copy()
        h, w = sub.shape

        # Hole indices in the target frame region.
        hole_id = np.full((h, w), -1, dtype=int)
        hole_coords = []
        for r in range(h):
            for c in range(w):
                if sub[r, c] == B:
                    hole_id[r, c] = len(hole_coords)
                    hole_coords.append((r, c))

        n_holes = len(hole_coords)
        if n_holes == 0:
            return sub

        full_mask = (1 << n_holes) - 1

        # Pieces are all cells that are neither frame nor empty color.
        piece_mask = ~((grid == F) | (grid == B))

        # Pieces should not already be inside this exact two-color region.
        if piece_mask[r1:r2 + 1, c1:c2 + 1].any():
            return None

        piece_components = get_components(piece_mask)
        if not piece_components:
            return None

        # Normalize each component into a translation-invariant pattern.
        individual_pieces = []
        for comp in piece_components:
            min_r = min(r for r, _ in comp)
            min_c = min(c for _, c in comp)
            cells = []
            for r, c in comp:
                cells.append((r - min_r, c - min_c, int(grid[r, c])))
            cells.sort()
            individual_pieces.append({
                "key": tuple(cells),
                "area": len(cells),
                "orig": (min_r, min_c),
            })

        total_piece_area = sum(p["area"] for p in individual_pieces)
        if total_piece_area < n_holes:
            return None

        # Group identical patterns.
        key_to_idx = {}
        pattern_keys = []
        origs_by_pattern = {}

        for p in individual_pieces:
            k = p["key"]
            if k not in key_to_idx:
                key_to_idx[k] = len(pattern_keys)
                pattern_keys.append(k)
                origs_by_pattern[k] = []
            origs_by_pattern[k].append(p["orig"])

        n_patterns = len(pattern_keys)
        pattern_area = [len(k) for k in pattern_keys]
        pattern_origs = [origs_by_pattern[k] for k in pattern_keys]

        # Precompute all legal placements for each pattern.
        placements = []
        placement_masks = []
        single_placement_by_hole = []
        hole_cands = [[] for _ in range(n_holes)]

        for pi, key in enumerate(pattern_keys):
            cells = list(key)
            ph = max(dr for dr, _, _ in cells) + 1
            pw = max(dc for _, dc, _ in cells) + 1

            pls = []

            if ph <= h and pw <= w:
                for tr in range(h - ph + 1):
                    for tc in range(w - pw + 1):
                        mask = 0
                        pairs = []
                        ok = True

                        for dr, dc, col in cells:
                            hid = hole_id[tr + dr, tc + dc]
                            if hid == -1:
                                ok = False
                                break
                            mask |= (1 << hid)
                            pairs.append((hid, col))

                        if ok:
                            pls.append((mask, pairs, tr, tc))

            placements.append(pls)
            placement_masks.append([p[0] for p in pls])

            if len(cells) == 1:
                d = {}
                for pl_idx, (_, pairs, _, _) in enumerate(pls):
                    d[pairs[0][0]] = pl_idx
                single_placement_by_hole.append(d)
            else:
                single_placement_by_hole.append(None)

            for pl_idx, (mask, _, _, _) in enumerate(pls):
                m = mask
                while m:
                    low = m & -m
                    hid = low.bit_length() - 1
                    hole_cands[hid].append((pi, pl_idx))
                    m ^= low

        # A pattern that cannot be placed at all cannot be kept.
        pattern_fits = [len(pls) > 0 for pls in placements]

        # Choose which individual pieces to keep if there is excess area.
        keepable_area = sum(p["area"] for p, ok in zip(individual_pieces, pattern_fits) if ok)
        if keepable_area < n_holes:
            return None

        kept = None
        if keepable_area == n_holes:
            kept = [ok for ok in pattern_fits]
        else:
            # DP over individual pieces to choose a subset with total area n_holes.
            # Higher score = more desirable to keep.
            n_p = len(individual_pieces)
            scores = [None] * n_p

            for i, p in enumerate(individual_pieces):
                pi = key_to_idx[p["key"]]
                if not pattern_fits[pi]:
                    scores[i] = None
                    continue
                pc = len(placements[pi])
                has_zero = any(col == 0 for _, _, col in p["key"])
                # Fixed total area makes the area term constant, but it is harmless.
                score = (
                    100000
                    - min(99999, pc)          # constrained pieces are preferred
                    + (1000 if not has_zero else 0)
                    + min(1000, p["area"])
                )
                scores[i] = score

            neg = -10**18
            dp = [neg] * (n_holes + 1)
            parent = [None] * (n_holes + 1)
            dp[0] = 0

            for i in range(n_p):
                if scores[i] is None:
                    continue
                a = individual_pieces[i]["area"]
                if a > n_holes:
                    continue
                for target in range(n_holes, a - 1, -1):
                    if dp[target - a] == neg:
                        continue
                    val = dp[target - a] + scores[i]
                    if val > dp[target]:
                        dp[target] = val
                        parent[target] = (target - a, i)

            if dp[n_holes] == neg:
                return None

            kept = [False] * n_p
            target = n_holes
            while target > 0:
                prev, i = parent[target]
                kept[i] = True
                target = prev

        rem = [0] * n_patterns
        for i, k in enumerate(kept):
            if k:
                pi = key_to_idx[individual_pieces[i]["key"]]
                rem[pi] += 1

        # Sanity: selected pieces must exactly cover all holes.
        if sum(rem[pi] * pattern_area[pi] for pi in range(n_patterns)) != n_holes:
            return None

        active_types = [pi for pi in range(n_patterns) if rem[pi] > 0]

        # ------------------------------------------------------------------
        # Backtracking exact cover.
        # ------------------------------------------------------------------
        solution = []
        failed = set()
        nodes = [0]
        NODE_LIMIT = 500000

        def distance_for_pattern(pi, tr, tc):
            # Only for ordering branches.
            best = 10**9
            for r0, c0 in pattern_origs[pi][:4]:  # enough for a heuristic
                d = abs(tr - r0) + abs(tc - c0)
                if d < best:
                    best = d
            return best

        def feasible(occ, rem_tuple):
            # Quick individual availability check.
            uncovered = full_mask ^ occ
            unc_count = uncovered.bit_count()

            for pi in active_types:
                need = rem_tuple[pi]
                if need == 0:
                    continue

                if pattern_area[pi] == 1:
                    if need > unc_count:
                        return False
                    continue

                avail = 0
                for mask in placement_masks[pi]:
                    if (mask & occ) == 0:
                        avail += 1
                        if avail >= need:
                            break
                if avail < need:
                    return False

            return True

        def finish_all_singles(occ, rem_tuple):
            """If only 1-cell pieces remain, assign them without branching."""
            rem_list = list(rem_tuple)

            # Check whether every remaining piece is single-cell.
            for pi in active_types:
                if rem_list[pi] > 0 and pattern_area[pi] != 1:
                    return False

            # Remaining holes in row-major order.
            rem_holes = []
            m = full_mask ^ occ
            while m:
                low = m & -m
                hid = low.bit_length() - 1
                rem_holes.append(hid)
                m ^= low

            # Build remaining single instances.
            instances = []
            for pi in active_types:
                cnt = rem_list[pi]
                if cnt == 0:
                    continue
                color = pattern_keys[pi][0][2]
                orig = pattern_origs[pi][0] if pattern_origs[pi] else (0, 0)
                for _ in range(cnt):
                    instances.append((pi, color, orig))

            if len(instances) != len(rem_holes):
                return False

            # If there are different colors, use a greedy nearest assignment.
            # For identical colors this is irrelevant.
            available = list(range(len(instances)))
            assignments = []

            for hid in rem_holes:
                hr, hc = hole_coords[hid]
                best_idx = -1
                best_score = None

                for idx in available:
                    pi, col, (r0, c0) = instances[idx]
                    d = abs(hr - r0) + abs(hc - c0)
                    # Tie-break deterministically.
                    score = (d, r0, c0, pi)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = idx

                if best_idx == -1:
                    return False

                pi = instances[best_idx][0]
                pl = single_placement_by_hole[pi].get(hid)
                if pl is None:
                    return False

                assignments.append((pi, pl))
                available.remove(best_idx)

            solution.extend(assignments)
            return True

        def dfs(occ, rem_tuple):
            nodes[0] += 1
            if nodes[0] > NODE_LIMIT:
                return False

            if occ == full_mask:
                return True

            state = (occ, rem_tuple)
            if state in failed:
                return False

            if not feasible(occ, rem_tuple):
                failed.add(state)
                return False

            # Special-case the final single-cell phase.
            if finish_all_singles(occ, rem_tuple):
                return True

            # MRV: choose uncovered hole with fewest possible placements.
            m = full_mask ^ occ
            best_cnt = 10**9
            best_cands = None

            while m:
                low = m & -m
                hid = low.bit_length() - 1
                m ^= low

                cnt = 0
                cands = []

                for pi, pl in hole_cands[hid]:
                    if rem_tuple[pi] == 0:
                        continue
                    if placements[pi][pl][0] & occ:
                        continue

                    cnt += 1
                    if cnt <= best_cnt:
                        cands.append((pi, pl))
                    if cnt > best_cnt:
                        break

                if cnt == 0:
                    failed.add(state)
                    return False

                if cnt < best_cnt:
                    best_cnt = cnt
                    best_cands = cands
                    if cnt == 1:
                        break

            if best_cands is None:
                failed.add(state)
                return False

            # Branch ordering:
            # 1) more constrained pattern first
            # 2) larger area first
            # 3) closer to original position first
            def cand_key(item):
                pi, pl = item
                tr, tc = placements[pi][pl][2], placements[pi][pl][3]
                d = distance_for_pattern(pi, tr, tc)
                return (
                    len(placements[pi]),
                    -pattern_area[pi],
                    d,
                    tr,
                    tc
                )

            best_cands.sort(key=cand_key)
            rem_list = list(rem_tuple)

            for pi, pl in best_cands:
                mask = placements[pi][pl][0]
                if mask & occ:
                    continue
                if rem_list[pi] == 0:
                    continue

                rem_list[pi] -= 1
                solution.append((pi, pl))

                if dfs(occ | mask, tuple(rem_list)):
                    return True

                solution.pop()
                rem_list[pi] += 1

            failed.add(state)
            return False

        if not dfs(0, tuple(rem)):
            return None

        # Apply the found placements.
        out = sub.copy()
        for pi, pl in solution:
            _, pairs, _, _ = placements[pi][pl]
            for hid, col in pairs:
                r, c = hole_coords[hid]
                out[r, c] = col

        # Verify that all holes are filled.
        for r, c in hole_coords:
            if out[r, c] == B:
                return None

        return out

    # Try candidates in preferred order.
    last_sub = None
    for score_tuple in candidates:
        _, _, _, F, B, r1, r2, c1, c2 = score_tuple
        res = solve_candidate(F, B, r1, r2, c1, c2)
        if res is not None:
            return res

        # Keep the best candidate's raw frame as a fallback shape.
        if last_sub is None:
            last_sub = grid[r1:r2 + 1, c1:c2 + 1].copy()

    if last_sub is not None:
        return last_sub

    return grid.copy()