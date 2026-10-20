import numpy as np
import sys

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble a framed puzzle.

    The input contains:
      - a "frame" color F that forms one connected component;
      - a second color B that occupies the empty cells inside the bounding
        box of that frame component;
      - all other colors are pieces.

    The output is the bounding box of that frame component, where every B
    cell is filled by the pieces.  Pieces are kept exactly as they appear
    (no rotation/reflection) and are packed into the empty cells.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.size == 0:
        return grid.copy()

    sys.setrecursionlimit(10000)

    def get_components(mask: np.ndarray):
        """Return a list of 4-connected components as lists of (r, c)."""
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

    colors = [int(c) for c in np.unique(grid)]
    piece_count_cache = {}

    def piece_count(F, B):
        key = (F, B)
        if key not in piece_count_cache:
            piece_count_cache[key] = int(np.sum(~((grid == F) | (grid == B))))
        return piece_count_cache[key]

    candidates = []

    # Find all possible frame components.
    for F in colors:
        mask_f = grid == F
        if not mask_f.any():
            continue

        for comp in get_components(mask_f):
            rs = [r for r, _ in comp]
            cs = [c for _, c in comp]
            r1, r2 = min(rs), max(rs)
            c1, c2 = min(cs), max(cs)

            sub = grid[r1:r2 + 1, c1:c2 + 1]
            uniq = [int(x) for x in np.unique(sub)]

            # A valid frame component must contain exactly two colors:
            # the frame color F and the empty-cell color B.
            if len(uniq) != 2:
                continue

            B = F if F != uniq[0] else uniq[1]
            if B == F:
                continue
            if F not in uniq or B not in uniq:
                continue

            holes = int(np.sum(sub != F))
            if holes == 0:
                continue

            pc = piece_count(F, B)
            if pc != holes:
                continue

            # Larger candidate first; component area as secondary tie-breaker.
            candidates.append((holes, len(comp), F, B, r1, r2, c1, c2))

    candidates.sort(reverse=True)

    def solve_candidate(F, B, r1, r2, c1, c2):
        h = r2 - r1 + 1
        w = c2 - c1 + 1

        frame = (grid[r1:r2 + 1, c1:c2 + 1] == F)
        hole_id = np.full((h, w), -1, dtype=int)
        hole_coords = []

        for r in range(h):
            for c in range(w):
                if not frame[r, c]:
                    hole_id[r, c] = len(hole_coords)
                    hole_coords.append((r, c))

        n_holes = len(hole_coords)

        if n_holes == 0:
            out = np.zeros((h, w), dtype=grid.dtype)
            out[frame] = F
            return out

        piece_mask = ~((grid == F) | (grid == B))
        piece_comps = get_components(piece_mask)

        # Normalize pieces into translation-invariant patterns.
        patterns_dict = {}
        for comp in piece_comps:
            min_r = min(r for r, _ in comp)
            min_c = min(c for _, c in comp)
            offsets = []
            for r, c in comp:
                offsets.append((r - min_r, c - min_c, int(grid[r, c])))

            signature = tuple(sorted(offsets))
            if signature not in patterns_dict:
                ph = max(dr for dr, _, _ in offsets) + 1
                pw = max(dc for _, dc, _ in offsets) + 1
                patterns_dict[signature] = {
                    "count": 0,
                    "offsets": offsets,
                    "ph": ph,
                    "pw": pw,
                }
            patterns_dict[signature]["count"] += 1

        patterns = list(patterns_dict.values())
        p_count = [p["count"] for p in patterns]
        p_areas = [len(p["offsets"]) for p in patterns]
        n_patterns = len(patterns)

        if sum(p_count) == 0:
            return None

        # Precompute all legal placements of each pattern.
        placements = []
        hole_cands = [[] for _ in range(n_holes)]
        full_mask = (1 << n_holes) - 1

        for pi, pat in enumerate(patterns):
            ph, pw = pat["ph"], pat["pw"]
            offs = pat["offsets"]
            pat_placements = []

            if ph > h or pw > w:
                placements.append([])
                continue

            for tr in range(h - ph + 1):
                for tc in range(w - pw + 1):
                    mask = 0
                    pairs = []
                    ok = True

                    for dr, dc, col in offs:
                        hr, hc = tr + dr, tc + dc
                        hid = hole_id[hr, hc]
                        if hid == -1:
                            ok = False
                            break
                        mask |= (1 << hid)
                        pairs.append((hid, col))

                    if ok:
                        pat_placements.append((mask, tuple(pairs)))

            placements.append(pat_placements)

            if p_count[pi] > 0 and not pat_placements:
                return None

            for pl, (mask, pairs) in enumerate(pat_placements):
                for hid, _ in pairs:
                    hole_cands[hid].append((pi, pl))

        # Basic necessary prune: enough non-overlapping-with-current
        # placement slots must exist for each remaining pattern type.
        for pi in range(n_patterns):
            if p_count[pi] > len(placements[pi]):
                return None

        nodes = [0]
        NODE_LIMIT = 200000
        assigned = []

        def dfs(occ_mask, rem):
            nodes[0] += 1
            if nodes[0] > NODE_LIMIT:
                return False

            if occ_mask == full_mask:
                return True

            # Prune: each remaining pattern must have at least rem[p]
            # currently available placements.
            for pi in range(n_patterns):
                if rem[pi] == 0:
                    continue

                avail = 0
                for mask, _ in placements[pi]:
                    if (mask & occ_mask) == 0:
                        avail += 1
                        if avail >= rem[pi]:
                            break

                if avail < rem[pi]:
                    return False

            # Choose an uncovered hole with the fewest candidate placements.
            best_cnt = 10**9
            best_cands = None

            m = full_mask ^ occ_mask
            while m:
                low = m & -m
                hid = low.bit_length() - 1
                m ^= low

                cnt = 0
                cands = []

                for pi, pl in hole_cands[hid]:
                    if rem[pi] == 0:
                        continue

                    pmask = placements[pi][pl][0]
                    if pmask & occ_mask:
                        continue

                    cnt += 1
                    if cnt <= best_cnt:
                        cands.append((pi, pl))

                    if cnt > best_cnt:
                        break

                if cnt == 0:
                    return False

                if cnt < best_cnt:
                    best_cnt = cnt
                    best_cands = cands
                    if cnt == 1:
                        break

            # Branch larger pieces first; this avoids wasting time on
            # unconstrained single cells when possible.
            best_cands.sort(key=lambda x: (-p_areas[x[0]], x[0], x[1]))

            for pi, pl in best_cands:
                pmask, pairs = placements[pi][pl]
                rem[pi] -= 1
                assigned.append((pi, pl))

                if dfs(occ_mask | pmask, rem):
                    return True

                assigned.pop()
                rem[pi] += 1

            return False

        rem = p_count[:]
        if not dfs(0, rem):
            return None

        out = np.zeros((h, w), dtype=grid.dtype)
        out[frame] = F

        for pi, pl in assigned:
            _, pairs = placements[pi][pl]
            for hid, col in pairs:
                r, c = hole_coords[hid]
                out[r, c] = col

        return out

    for holes, area, F, B, r1, r2, c1, c2 in candidates:
        result = solve_candidate(F, B, r1, r2, c1, c2)
        if result is not None:
            return result

    # Fallback: should not be needed for valid ARC instances.
    return grid.copy()