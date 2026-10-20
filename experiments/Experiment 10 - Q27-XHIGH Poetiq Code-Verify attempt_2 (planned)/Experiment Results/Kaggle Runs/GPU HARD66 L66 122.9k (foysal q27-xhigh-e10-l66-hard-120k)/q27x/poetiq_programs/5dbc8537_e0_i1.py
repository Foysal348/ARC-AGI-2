import numpy as np
from itertools import combinations


def transform(grid: np.ndarray) -> np.ndarray:
    """
    General solver for the 'fill the board holes with pieces from the other side'
    ARC task family.

    The input grid is expected to contain two panels separated by a uniform line
    (possibly two adjacent uniform lines). One panel is a board with holes,
    the other panel contains pieces on a background color. The output is the
    board panel with the hole cells filled by the pieces.

    The solver is intentionally general:
      * tries all plausible splits,
      * tries plausible hole/background colors,
      * extracts pieces as connected non-background components,
      * solves an exact-cover tiling with backtracking,
      * falls back to rotations and a greedy fill if necessary.
    """
    grid = np.asarray(grid, dtype=int)
    H, W = grid.shape
    if H == 0 or W == 0:
        return grid.copy()

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def top_colors(arr, k=5):
        if arr.size == 0:
            return []
        vals, cnts = np.unique(arr, return_counts=True)
        order = np.argsort(cnts)[::-1]
        return [(int(vals[i]), int(cnts[i])) for i in order[:k]]

    def row_uniform(row):
        return len(set(row.tolist())) == 1

    def col_uniform(col):
        return len(set(col.tolist())) == 1

    def unique_list(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # ------------------------------------------------------------------
    # Split detection
    # ------------------------------------------------------------------
    splits = []

    # Single uniform horizontal separator row.
    for r in range(1, H - 1):
        if row_uniform(grid[r]):
            c = int(grid[r, 0])
            splits.append((1, 'h', r, r, [c], 'single'))

    # Two adjacent uniform horizontal rows.
    for r in range(1, H - 1):
        if row_uniform(grid[r]) and row_uniform(grid[r + 1]):
            c1 = int(grid[r, 0])
            c2 = int(grid[r + 1, 0])
            score = 0 if c1 != c2 else 2
            if r + 1 < H - 1:
                splits.append((score, 'h', r, r + 1, [c1, c2], 'double'))

    # Single uniform vertical separator column.
    for c in range(1, W - 1):
        if col_uniform(grid[:, c]):
            val = int(grid[0, c])
            splits.append((1, 'v', c, c, [val], 'single'))

    # Two adjacent uniform vertical columns.
    for c in range(1, W - 1):
        if col_uniform(grid[:, c]) and col_uniform(grid[:, c + 1]):
            v1 = int(grid[0, c])
            v2 = int(grid[0, c + 1])
            score = 0 if v1 != v2 else 2
            if c + 1 < W - 1:
                splits.append((score, 'v', c, c + 1, [v1, v2], 'double'))

    splits.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Piece extraction
    # ------------------------------------------------------------------
    def get_pieces(panel, bg):
        """
        Extract 4-connected components of non-background cells.
        Each piece is returned as:
            (colors_2d, mask_2d, area, cells)
        where cells is [(dr, dc, color), ...]
        """
        h, w = panel.shape
        visited = np.zeros((h, w), dtype=bool)
        pieces = []

        for i in range(h):
            for j in range(w):
                if visited[i, j] or panel[i, j] == bg:
                    continue

                stack = [(i, j)]
                visited[i, j] = True
                cells = []
                min_r = max_r = i
                min_c = max_c = j

                while stack:
                    r, c = stack.pop()
                    cells.append((r, c))
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if not visited[nr, nc] and panel[nr, nc] != bg:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                sub = panel[min_r:max_r + 1, min_c:max_c + 1].astype(int, copy=True)
                mask = (sub != bg)
                area = int(mask.sum())
                if area == 0:
                    continue

                vcells = []
                for r in range(sub.shape[0]):
                    for c in range(sub.shape[1]):
                        if mask[r, c]:
                            vcells.append((r, c, int(sub[r, c])))

                pieces.append((sub, mask, area, vcells))

        return pieces

    # ------------------------------------------------------------------
    # Piece variants (optional rotations/reflections)
    # ------------------------------------------------------------------
    def make_variants(piece, allow_rot=False, allow_refl=False):
        """
        Return a list of variants:
            (cells, orientation_score, height, width)
        where cells = [(dr, dc, color), ...]
        """
        base_arr = piece[0].astype(int, copy=True)
        base_mask = piece[1].astype(bool, copy=True)
        variants = []
        seen = set()

        for k in range(4 if allow_rot else 1):
            refl_options = [True, False] if allow_refl else [False]
            for refl in refl_options:
                arr = base_arr.copy()
                m = base_mask.copy()

                if refl:
                    arr = np.fliplr(arr)
                    m = np.fliplr(m)
                if k:
                    arr = np.rot90(arr, k)
                    m = np.rot90(m, k)

                h, w = arr.shape
                vcells = []
                for i in range(h):
                    for j in range(w):
                        if m[i, j]:
                            vcells.append((i, j, int(arr[i, j])))

                if not vcells:
                    continue

                key = (
                    tuple(m.ravel().tolist()),
                    tuple(int(x) for x in arr[m].ravel())
                )
                if key in seen:
                    continue
                seen.add(key)

                if k == 0 and not refl:
                    score = 0
                elif refl:
                    score = 3
                elif k == 2:
                    score = 2
                else:
                    score = 1

                height = max(dr for dr, dc, _ in vcells) + 1
                width = max(dc for dr, dc, _ in vcells) + 1
                variants.append((vcells, score, height, width))

        if not variants:
            # Should not happen for valid pieces, but keep a safe fallback.
            cells = piece[3]
            height = max(dr for dr, dc, _ in cells) + 1
            width = max(dc for dr, dc, _ in cells) + 1
            variants = [(cells, 0, height, width)]

        return variants

    # ------------------------------------------------------------------
    # Exact cover / backtracking tiling solver
    # ------------------------------------------------------------------
    def exact_cover(board_mask, pieces, allowed_indices, require_all,
                    allow_rot=False, allow_refl=False, node_limit=100000):
        """
        Place pieces onto True cells of board_mask without overlap.
        If require_all is True, every allowed piece must be used.
        Returns a list of (r, c, cells) placements, or None.
        """
        hb, wb = board_mask.shape
        all_cells = []
        index = {}
        for r in range(hb):
            for c in range(wb):
                if board_mask[r, c]:
                    index[(r, c)] = len(all_cells)
                    all_cells.append((r, c))

        N = len(all_cells)
        if N == 0:
            return []

        all_mask = (1 << N) - 1
        allowed_count = len(allowed_indices)
        variants_by_piece = []
        area_by_pos = []

        for pi in allowed_indices:
            variants_by_piece.append(make_variants(pieces[pi], allow_rot, allow_refl))
            area_by_pos.append(pieces[pi][2])

        placements = []
        pos_has_placement = [False] * allowed_count

        for pos, pi in enumerate(allowed_indices):
            for vi, (vcells, score, height, width) in enumerate(variants_by_piece[pos]):
                if height > hb or width > wb:
                    continue

                for r in range(hb - height + 1):
                    for c in range(wb - width + 1):
                        pmask = 0
                        idxs = []
                        ok = True

                        for dr, dc, _ in vcells:
                            rr, cc = r + dr, c + dc
                            if not board_mask[rr, cc]:
                                ok = False
                                break
                            idx = index[(rr, cc)]
                            pmask |= (1 << idx)
                            idxs.append(idx)

                        if ok:
                            pid = len(placements)
                            placements.append((pos, pmask, r, c, vi, score, idxs))
                            pos_has_placement[pos] = True

        if not placements:
            return None

        cell_to_placements = [[] for _ in range(N)]
        for pid, placement in enumerate(placements):
            for idx in placement[6]:
                cell_to_placements[idx].append(pid)

        if any(len(x) == 0 for x in cell_to_placements):
            return None

        if require_all and not all(pos_has_placement):
            return None

        # Static heuristic: sum of option counts of cells covered by placement.
        cost = []
        for placement in placements:
            s = 0
            for idx in placement[6]:
                s += len(cell_to_placements[idx])
            cost.append(s)

        used = [False] * allowed_count
        nodes = [0]

        def rec(covered, used_count):
            nodes[0] += 1
            if nodes[0] > node_limit:
                return None

            if covered == all_mask:
                if require_all:
                    if used_count == allowed_count:
                        return []
                    return None
                return []

            unc = all_mask ^ covered
            best_count = 10**9
            best_cands = None

            u = unc
            while u:
                lsb = u & -u
                idx = lsb.bit_length() - 1
                u ^= lsb

                count = 0
                cands = []
                limit = best_count

                for pid in cell_to_placements[idx]:
                    pos, pmask, r, c, vi, score, idxs = placements[pid]
                    if not used[pos] and (pmask & covered) == 0:
                        count += 1
                        if count <= limit:
                            cands.append(pid)
                        if count >= limit:
                            break

                if count < best_count:
                    best_count = count
                    best_cands = cands
                    if count == 0:
                        return None
                    if count == 1:
                        break

            if best_cands is None:
                return None

            best_cands.sort(
                key=lambda pid: (
                    cost[pid],
                    -area_by_pos[placements[pid][0]],
                    placements[pid][5],
                    placements[pid][0]
                )
            )

            for pid in best_cands:
                pos, pmask, r, c, vi, score, idxs = placements[pid]
                if used[pos] or (pmask & covered) != 0:
                    continue

                used[pos] = True
                res = rec(covered | pmask, used_count + 1)
                if res is not None:
                    res.append((r, c, variants_by_piece[pos][vi][0]))
                    return res
                used[pos] = False

            return None

        return rec(0, 0)

    # ------------------------------------------------------------------
    # Try to solve one candidate mask with piece exclusion combos
    # ------------------------------------------------------------------
    def solve_task(mask, pieces, diff, allow_rot=False, allow_refl=False, limit=100000):
        n = len(pieces)
        if n == 0:
            return None

        areas = [p[2] for p in pieces]
        total_area = sum(areas)
        mask_area = int(mask.sum())
        if total_area < mask_area:
            return None

        # If exact area match, use all pieces.
        if diff == 0:
            allowed = list(range(n))
            sol = exact_cover(
                mask, pieces, allowed,
                require_all=True,
                allow_rot=allow_rot,
                allow_refl=allow_refl,
                node_limit=limit
            )
            if sol is not None:
                return sol
            return None

        # Try to exclude a small set of decoy pieces whose area sums to diff.
        max_count = 3 if diff <= 30 else 1
        max_combos = 20 if diff <= 30 else 5
        combos = []

        for k in range(1, min(max_count, n) + 1):
            for combo in combinations(range(n), k):
                s = 0
                for i in combo:
                    s += areas[i]
                    if s > diff:
                        break
                if s == diff:
                    combos.append(combo)
                if len(combos) >= max_combos:
                    break
            if len(combos) >= max_combos:
                break

        combos.sort(key=lambda c: (len(c), -max(areas[i] for i in c)))

        for combo in combos:
            excluded = set(combo)
            allowed = [i for i in range(n) if i not in excluded]
            if not allowed:
                continue

            allowed_area = sum(areas[i] for i in allowed)
            if allowed_area != mask_area:
                continue

            sol = exact_cover(
                mask, pieces, allowed,
                require_all=True,
                allow_rot=allow_rot,
                allow_refl=allow_refl,
                node_limit=limit
            )
            if sol is not None:
                return sol

        # Fallback: allow unused pieces and just cover the mask.
        sol = exact_cover(
            mask, pieces, list(range(n)),
            require_all=False,
            allow_rot=allow_rot,
            allow_refl=allow_refl,
            node_limit=max(10000, limit // 4)
        )
        return sol

    # ------------------------------------------------------------------
    # Greedy fallback if exact solving fails
    # ------------------------------------------------------------------
    def greedy_fill(board, mask, pieces):
        out = board.copy()
        hb, wb = mask.shape
        covered = np.zeros((hb, wb), dtype=bool)

        order = sorted(range(len(pieces)), key=lambda i: -pieces[i][2])

        for pi in order:
            piece = pieces[pi]
            placed = False

            for vcells, _, height, width in make_variants(piece, False, False):
                if height > hb or width > wb:
                    continue

                for r in range(hb - height + 1):
                    for c in range(wb - width + 1):
                        ok = True
                        for dr, dc, _ in vcells:
                            rr, cc = r + dr, c + dc
                            if not mask[rr, cc] or covered[rr, cc]:
                                ok = False
                                break
                        if ok:
                            for dr, dc, col in vcells:
                                rr, cc = r + dr, c + dc
                                out[rr, cc] = col
                                covered[rr, cc] = True
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

        return out

    # ------------------------------------------------------------------
    # Main search over splits, orientations, colors, masks
    # ------------------------------------------------------------------
    fallback_task = None  # (priority, board, mask, h, b, pieces)

    def node_limit_for(area):
        if area > 200:
            return 150000
        if area > 100:
            return 120000
        if area > 50:
            return 80000
        return 50000

    # Pass 1: translations only.
    # Pass 2: translations + rotations.
    for allow_rot in (False, True):
        for split in splits:
            _, orient, a, b, sep_colors, kind = split

            if orient == 'h':
                if kind == 'single':
                    A = grid[:a]
                    B = grid[a + 1:]
                    sep_A = sep_B = int(sep_colors[0])
                else:
                    A = grid[:a]
                    B = grid[b + 1:]
                    sep_A = int(sep_colors[0])
                    sep_B = int(sep_colors[1])
            else:
                if kind == 'single':
                    A = grid[:, :a]
                    B = grid[:, a + 1:]
                    sep_A = sep_B = int(sep_colors[0])
                else:
                    A = grid[:, :a]
                    B = grid[:, b + 1:]
                    sep_A = int(sep_colors[0])
                    sep_B = int(sep_colors[1])

            if A.size == 0 or B.size == 0:
                continue

            orientations = [
                (A, B, sep_A, sep_B),
                (B, A, sep_B, sep_A),
            ]

            for board, tray, board_sep, tray_sep in orientations:
                if board.size == 0 or tray.size == 0:
                    continue

                board_colors = top_colors(board, 6)
                if not board_colors:
                    continue
                tray_colors = top_colors(tray, 3)
                if not tray_colors:
                    continue

                most_board = board_colors[0][0]
                most_tray = tray_colors[0][0]

                h_cands = unique_list([
                    board_sep,
                    tray_sep,
                    most_board,
                    board_colors[1][0] if len(board_colors) > 1 else -1,
                    board_colors[2][0] if len(board_colors) > 2 else -1,
                ])

                b_cands = unique_list([
                    most_tray,
                    tray_sep,
                    board_sep,
                ])

                tasks = []

                for bg in b_cands:
                    pieces = get_pieces(tray, bg)
                    if not pieces:
                        continue
                    total_piece_area = sum(p[2] for p in pieces)

                    for hole in h_cands:
                        if hole < 0:
                            continue

                        full_mask = (board == hole)
                        full_area = int(full_mask.sum())
                        if full_area == 0 or total_piece_area < full_area:
                            continue

                        diff = total_piece_area - full_area
                        tasks.append((
                            abs(diff), 0, hole, bg, full_mask, diff, pieces, total_piece_area
                        ))

                        # Interior mask: ignore boundary cells. This is a robust
                        # fallback for boards where the outer ring is not a hole.
                        int_mask = full_mask.copy()
                        if int_mask.shape[0] > 1:
                            int_mask[0, :] = False
                            int_mask[-1, :] = False
                        if int_mask.shape[1] > 1:
                            int_mask[:, 0] = False
                            int_mask[:, -1] = False

                        int_area = int(int_mask.sum())
                        if int_area > 0 and total_piece_area >= int_area:
                            diff_int = total_piece_area - int_area
                            tasks.append((
                                abs(diff_int), 1, hole, bg, int_mask, diff_int, pieces, total_piece_area
                            ))

                tasks.sort(key=lambda x: (x[0], x[1], x[5]))

                # Limit work per orientation to keep runtime sane.
                for task in tasks[:12]:
                    _, _, hole, bg, mask, diff, pieces, total_area = task
                    area = int(mask.sum())

                    if fallback_task is None or (abs(diff), 0) < fallback_task[0]:
                        fallback_task = ((abs(diff), 0), board, mask, hole, bg, pieces)

                    lim = node_limit_for(area)
                    if allow_rot:
                        lim = max(20000, lim // 2)

                    sol = solve_task(
                        mask, pieces, diff,
                        allow_rot=allow_rot,
                        allow_refl=False,
                        limit=lim
                    )

                    if sol is not None:
                        out = board.copy()
                        for r, c, cells in sol:
                            for dr, dc, col in cells:
                                out[r + dr, c + dc] = col
                        return out.astype(int, copy=False)

    # ------------------------------------------------------------------
    # Final fallback: greedy fill on the best-looking board, otherwise return input.
    # ------------------------------------------------------------------
    if fallback_task is not None:
        _, board, mask, hole, bg, pieces = fallback_task
        return greedy_fill(board, mask, pieces).astype(int, copy=False)

    return grid.copy()