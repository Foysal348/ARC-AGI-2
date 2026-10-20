import numpy as np
from itertools import product

def _to_binary(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Convert the two-color grid to a binary grid using the smaller color as 0
    and the larger color as 1.  The transformation itself is color-agnostic
    up to this binary reinterpretation.
    """
    colors = sorted({int(x) for x in grid.ravel() if x != -1})
    if len(colors) == 1:
        c0 = colors[0]
        c1 = colors[0]
    elif len(colors) == 2:
        c0, c1 = colors
    else:
        # Fallback: treat 0 as background and non-zero as foreground.
        c0 = 0
        c1 = int(grid[grid != 0].mean())

    b = (grid == c1).astype(np.uint8)
    return b, c0, c1


def _from_binary(b: np.ndarray, c0: int, c1: int) -> np.ndarray:
    return np.where(b.astype(bool), c1, c0)


def _row_derivative(row: np.ndarray) -> int:
    """
    Encode the horizontal derivative of a 20-bit row as a 19-bit integer.
    derivative bit j is row[j+1] XOR row[j].
    """
    row = np.asarray(row, dtype=np.uint8)
    d = row[1:] ^ row[:-1]
    x = 0
    for bit in d:
        x = (x << 1) | int(bit)
    return x


def _expand_row(start_bit: int, der: int) -> np.ndarray:
    """
    Reconstruct a 20-bit row from its first bit and 19-bit derivative.
    """
    row = np.zeros(20, dtype=np.uint8)
    row[0] = start_bit & 1
    # derivative is stored as d[0] = bit1 XOR bit0, ..., d[18] = bit19 XOR bit18
    mask = 1
    for j in range(19, -1, -1):
        bit = (der >> j) & 1
        row[j + 1] = row[j] ^ bit
    # The mask variable is not needed, kept for clarity.
    return row


def _allowed_row_states(row_values: np.ndarray, left: int, width: int):
    """
    For one constrained output row, enumerate all possible (start_bit, derivative)
    pairs whose reconstructed 20-bit row contains the given segment.

    Returns a list of (derivative_int, start_bit).
    """
    states = []
    # If the fixed segment determines the derivative bits strictly inside it.
    # We simply brute force all 2^19 derivatives for correctness; 2^19 is small.
    fixed = row_values  # length width
    for start in (0, 1):
        for der in range(1 << 19):
            full = _expand_row(start, der)
            if np.array_equal(full[left:left + width], fixed):
                states.append((der, start))
    return states


def _min_hamming_path(states_per_row):
    """
    Given, for each input row in vertical order, a set of allowed derivative
    states, choose one derivative per row minimizing the sum of Hamming
    distances between consecutive rows.

    This is the core completion rule inferred from the examples:
    the completed 20x20 grid is the one that keeps the horizontal-derivative
    sequence as stable as possible while containing the input patch.
    """
    n = len(states_per_row)
    if n == 0:
        return [], []

    # Initial DP.
    prev = {}
    for der, start in states_per_row[0]:
        prev[der] = (0, start)  # cost, start bit

    path = [None] * n
    path[0] = list(states_per_row[0])

    for i in range(1, n):
        curr = {}
        for der2, start2 in states_per_row[i]:
            best_cost = None
            best_prev_der = None
            best_prev_start = None
            # Small state sets in the observed cases; direct DP is enough.
            for der1, (cost1, start1) in prev.items():
                cost = cost1 + bin(der1 ^ der2).count("1")
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_prev_der = der1
                    best_prev_start = start1
            curr[der2] = (best_cost, best_prev_der)
        prev = curr
        path[i] = curr

    # Recover a full path by storing parents.  Redo with parent tracking.
    parent = [{}]
    dp = {}
    for der, start in states_per_row[0]:
        dp[der] = (0, start)

    for i in range(1, n):
        new_dp = {}
        new_par = {}
        for der2, start2 in states_per_row[i]:
            best = None
            best_d1 = None
            for der1, (cost1, _) in dp.items():
                c = cost1 + bin(der1 ^ der2).count("1")
                if best is None or c < best:
                    best = c
                    best_d1 = der1
            new_dp[der2] = (best, start2)
            new_par[der2] = best_d1
        dp = new_dp
        parent.append(new_par)

    # Choose final derivative with minimum cost.
    final_der = min(dp.items(), key=lambda kv: kv[1][0])[0]
    ders = [None] * n
    starts = [None] * n
    ders[n - 1] = final_der
    starts[n - 1] = dp[final_der][1]
    for i in range(n - 2, -1, -1):
        final_der = parent[i + 1][final_der]
        ders[i] = final_der
        starts[i] = dp[final_der][1]

    return ders, starts


def _fill_full_grid(input_binary: np.ndarray, top: int, left: int):
    """
    Complete a 20x20 binary grid assuming the input is placed at (top, left).
    """
    H, W = input_binary.shape
    states_per_row = []
    for r in range(H):
        states_per_row.append(_allowed_row_states(input_binary[r], left, W))
        if not states_per_row[-1]:
            return None, None

    ders, starts = _min_hamming_path(states_per_row)
    if ders is None:
        return None, None

    full = np.zeros((20, 20), dtype=np.uint8)

    # Fill constrained rows exactly.
    for r in range(H):
        full[top + r, :] = _expand_row(starts[r], ders[r])

    # Fill unconstrained rows by propagating the nearest constrained derivative.
    constrained_rows = list(range(H))
    all_rows = list(range(20))
    row_der = {}
    row_start = {}
    for r, out_r in enumerate(range(top, top + H)):
        row_der[out_r] = ders[r]
        row_start[out_r] = starts[r]

    # For rows before the first constrained row, use the first constrained row.
    first_constrained_out = top
    for i in range(first_constrained_out):
        row_der[i] = row_der[first_constrained_out]
        row_start[i] = row_start[first_constrained_out]

    # For rows after the last constrained row, use the last constrained row.
    last_constrained_out = top + H - 1
    for i in range(last_constrained_out + 1, 20):
        row_der[i] = row_der[last_constrained_out]
        row_start[i] = row_start[last_constrained_out]

    for i in range(20):
        if i in row_der:
            full[i, :] = _expand_row(row_start[i], row_der[i])
        else:
            # Should not happen.
            full[i, :] = 0

    # Force the input patch exactly, in case of ties.
    full[top:top + H, left:left + W] = input_binary

    return full, None


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the 20x20 output grid from the given input patch.

    The transformation is binary and color-agnostic:
    - Map the two colors to 0/1 (smaller color -> 0).
    - Try to place the input patch in a 20x20 grid.
    - For a fixed placement, complete the remaining rows by choosing, for each
      constrained row, a 20-bit extension.  Each extension is represented by
      its first bit and its 19-bit horizontal derivative.
    - Choose the sequence of derivatives that minimizes the total number of
      horizontal-derivative changes between consecutive constrained rows.
    - Fill all other rows by continuing the nearest constrained row's
      derivative, then restore the input patch exactly.
    - Finally, map 0/1 back to the original two colors.
    """
    if grid.size == 0:
        return np.zeros_like(grid)

    b, c0, c1 = _to_binary(grid)
    H, W = b.shape

    # The observed task always expands to 20x20.
    N = 20

    # Try all valid placements.  In the examples, the correct placement is the
    # one with the smallest derivative-change cost.  Ties are broken toward the
    # top-left, which matches the provided samples.
    best_full = None
    best_cost = None
    best_pos = (10**9, 10**9)

    for top in range(0, N - H + 1):
        for left in range(0, N - W + 1):
            full, _ = _fill_full_grid(b, top, left)
            if full is None:
                continue

            # Score the placement by the actual derivative-change cost of the
            # completed full grid.  This selects the natural completion.
            cost = 0
            for i in range(1, N):
                d0 = _row_derivative(full[i - 1])
                d1 = _row_derivative(full[i])
                cost += bin(d0 ^ d1).count("1")

            # Small preference for placements touching the top/left edges,
            # consistent with the examples and giving deterministic behavior.
            pos_penalty = top * 0.1 + left * 0.1
            total = cost + pos_penalty

            if best_cost is None or total < best_cost:
                best_cost = total
                best_full = full.copy()
                best_pos = (top, left)

    if best_full is None:
        # Fallback: center/top-left placement and nearest-neighbor fill.
        top = max(0, (N - H) // 2)
        left = max(0, (N - W) // 2)
        best_full = np.zeros((N, N), dtype=np.uint8)
        best_full[top:top + H, left:left + W] = b
        # Very simple deterministic fill: replicate the input patch to cover.
        for i in range(N):
            for j in range(N):
                best_full[i, j] = b[min(i - top, H - 1) if 0 <= i - top < H else H - 1,
                                     min(j - left, W - 1) if 0 <= j - left < W else W - 1]

    return _from_binary(best_full, c0, c1)