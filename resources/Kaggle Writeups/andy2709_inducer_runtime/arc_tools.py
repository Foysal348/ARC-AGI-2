"""Tested ARC grid helpers. Grids are lists of rows of ints 0-9; cells are (row, col); bboxes are (r0, c0, r1, c1), inclusive.

load(path="puzzle.json")                  -> {"train": [{"input", "output"}], "test": [{"input"}]}
show(grid)                                -> print the grid with row and column indices
diff(a, b, limit=20)                      -> [(r, c, a_value, b_value)] (or a shape message); prints a summary
check(transform, puzzle=None)             -> True if transform reproduces every training output; prints each pair's result
compare(candidates, puzzle=None)          -> run {name: transform}; print training pairs each passes and where passing ones disagree on tests
write_predictions(transform, puzzle=None) -> run transform on every test input, validate and write predictions.json
background(grid)                          -> most common colour
background_report(puzzle=None)            -> prints each grid's most common colour; flags test inputs that differ from training
objects(grid, conn=4, bg=None, by_color=True)
                                          -> [Obj(color, cells, size, bbox, shape, touches_border)]; bg=None uses background(grid)
conn_diff(grid, bg=None, by_color=True)   -> 8-connected objects that split into several 4-connected ones
fragments(grid, color, gap=1, bg=None)    -> groups of same-colour objects at most `gap` empty cells apart
bbox(grid_or_cells, bg=None)              -> bbox of the non-background cells (or of the given cells)
crop(grid, box)                           -> the subgrid inside an inclusive bbox
lines(grid, bg=None)                      -> {"rows": {r: color}, "cols": {c: color}} for full one-colour lines (crossings allowed)
split_by_lines(grid, bg=None)             -> (line rows, line cols, blocks[i][j]) for the cells between full lines
ray(grid, start, step, stop="non_bg")     -> (cells, stop_cell, stop_color) from start moving by step=(dr, dc); stop: "non_bg", "edge" or colours
dihedral(grid)                            -> {name: grid} for the 8 rotations/reflections; find_dihedral(a, b) -> names mapping a to b
shape_rules(puzzle=None, limit=8)         -> output-height and -width rules that fit every training pair, with test predictions
"""
from __future__ import annotations

import json
from collections import Counter, namedtuple

Obj = namedtuple("Obj", "color cells size bbox shape touches_border")
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = _N4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def load(path="puzzle.json"):
    with open(path) as stream:
        return json.load(stream)


def _puzzle(puzzle):
    return load() if puzzle is None else puzzle


def show(grid):
    """Print the grid with row and column indices (columns modulo 10 above the first row)."""
    width = len(grid[0])
    print("    " + "".join(str(c % 10) for c in range(width)))
    for r, row in enumerate(grid):
        print(f"{r:>3} " + "".join(str(v) for v in row))


def diff(a, b, limit=20):
    """Cells where grids a and b differ, as (row, col, a_value, b_value); a string if the shapes differ."""
    if (len(a), len(a[0])) != (len(b), len(b[0])):
        message = f"shape {len(a)}x{len(a[0])} vs {len(b)}x{len(b[0])}"
        print(message)
        return message
    cells = [(r, c, a[r][c], b[r][c]) for r in range(len(a)) for c in range(len(a[0])) if a[r][c] != b[r][c]]
    print(f"{len(cells)} cells differ" + (": " + ", ".join(f"({r},{c}) {x}->{y}" for r, c, x, y in cells[:limit])
                                          + (" ..." if len(cells) > limit else "") if cells else ""))
    return cells


def _as_grid(value):
    value = value.tolist() if hasattr(value, "tolist") else value
    return [[int(v) for v in row] for row in value]


def _valid(grid):
    return (isinstance(grid, list) and 1 <= len(grid) <= 30 and isinstance(grid[0], list) and 1 <= len(grid[0]) <= 30
            and all(isinstance(row, list) and len(row) == len(grid[0]) and all(0 <= v <= 9 for v in row)
                    for row in grid))


def check(transform, puzzle=None):
    """Run transform on every training pair and print ok / differing cells / errors; also report test output shapes."""
    puzzle, passed = _puzzle(puzzle), True
    for index, pair in enumerate(puzzle["train"]):
        try:
            got = _as_grid(transform(pair["input"]))
        except Exception as exc:  # noqa: BLE001 - report any failure of the candidate program
            print(f"train {index}: {type(exc).__name__}: {exc}")
            passed = False
            continue
        if got == pair["output"]:
            print(f"train {index}: ok")
            continue
        passed = False
        print(f"train {index}: ", end="")
        diff(got, pair["output"], limit=8)
    shapes = []
    for test in puzzle["test"]:
        try:
            out = _as_grid(transform(test["input"]))
            shapes.append(f"{len(out)}x{len(out[0])}")
        except Exception as exc:  # noqa: BLE001
            shapes.append(f"{type(exc).__name__}")
    print(("all training pairs pass" if passed else "some training pairs fail") + "; test outputs: " + ", ".join(shapes))
    return passed


def compare(candidates, puzzle=None, limit=10):
    """Run several candidate transforms ({name: function}) on every pair; print which training pairs each passes, then,
    among the candidates passing all of them, whether their test outputs agree and where they differ."""
    puzzle, results = _puzzle(puzzle), {}
    for name, transform in candidates.items():
        passed, tests = [], []
        for index, pair in enumerate(puzzle["train"]):
            try:
                passed += [index] if _as_grid(transform(pair["input"])) == pair["output"] else []
            except Exception:  # noqa: BLE001
                pass
        for test in puzzle["test"]:
            try:
                tests.append(_as_grid(transform(test["input"])))
            except Exception as exc:  # noqa: BLE001
                tests.append(f"{type(exc).__name__}")
        results[name] = {"train_passed": passed, "test": tests}
        print(f"{name}: passes training pairs {passed} of {len(puzzle['train'])}")
    fitting = [name for name, result in results.items() if len(result["train_passed"]) == len(puzzle["train"])]
    for index in range(len(puzzle["test"])):
        outputs = {name: results[name]["test"][index] for name in fitting}
        distinct = {json.dumps(grid) for grid in outputs.values()}
        if len(distinct) <= 1:
            print(f"test {index}: " + ("all fitting candidates agree" if fitting else "no candidate fits training"))
            continue
        first = fitting[0]
        print(f"test {index}: {len(distinct)} different outputs among {len(fitting)} fitting candidates")
        for name in fitting[1:]:
            a, b = outputs[first], outputs[name]
            if isinstance(a, list) and isinstance(b, list) and a != b:
                print(f"  {first} vs {name}: ", end="")
                diff(a, b, limit=limit)
    return results


def write_predictions(transform, puzzle=None, path="predictions.json"):
    """Apply transform to every test input, check each output is a valid grid, write them in order, print shapes."""
    outputs = [_as_grid(transform(test["input"])) for test in _puzzle(puzzle)["test"]]
    bad = [i for i, grid in enumerate(outputs) if not _valid(grid)]
    if bad:
        raise ValueError(f"test outputs {bad} are not valid grids (1-30 rows of equal length, colours 0-9)")
    with open(path, "w") as stream:
        json.dump(outputs, stream)
    print(f"wrote {path}: " + ", ".join(f"{len(g)}x{len(g[0])}" for g in outputs))
    return outputs


def background(grid):
    return Counter(v for row in grid for v in row).most_common(1)[0][0]


def background_report(puzzle=None):
    """Print each grid's most common colour (with ties) and flag test inputs that differ from every training input."""
    puzzle = _puzzle(puzzle)

    def top(grid):
        counts = Counter(v for row in grid for v in row).most_common()
        return [color for color, n in counts if n == counts[0][1]]

    train = [top(pair["input"]) for pair in puzzle["train"]]
    print("train inputs: " + ", ".join(f"{t}" for t in train) + "; train outputs: "
          + ", ".join(f"{top(pair['output'])}" for pair in puzzle["train"]))
    seen = {color for colors in train for color in colors}
    for index, test in enumerate(puzzle["test"]):
        colors = top(test["input"])
        note = "" if set(colors) & seen else f"  <- differs from every training input ({sorted(seen)})"
        print(f"test {index} input: {colors}" + (" (tie)" if len(colors) > 1 else "") + note)


def _components(grid, keep, conn, same):
    height, width = len(grid), len(grid[0])
    steps = _N8 if conn == 8 else _N4
    seen, found = set(), []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen or not keep(grid[r][c]):
                continue
            stack, cells = [(r, c)], []
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in steps:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and keep(grid[ny][nx])
                            and same(grid[y][x], grid[ny][nx])):
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            found.append(sorted(cells))
    return found


def _obj(grid, cells):
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    colors = {grid[r][c] for r, c in cells}
    shape = tuple(tuple(int((r, c) in set(cells)) for c in range(c0, c1 + 1)) for r in range(r0, r1 + 1))
    border = r0 == 0 or c0 == 0 or r1 == len(grid) - 1 or c1 == len(grid[0]) - 1
    return Obj(colors.pop() if len(colors) == 1 else frozenset(colors), cells, len(cells), (r0, c0, r1, c1), shape,
               border)


def objects(grid, conn=4, bg=None, by_color=True):
    """Connected non-background objects. by_color=False joins touching cells of different colours (color is then a set)."""
    if conn not in (4, 8):
        raise ValueError("conn must be 4 or 8")
    bg = background(grid) if bg is None else bg
    same = (lambda a, b: a == b) if by_color else (lambda a, b: True)
    return [_obj(grid, cells) for cells in _components(grid, lambda v: v != bg, conn, same)]


def conn_diff(grid, bg=None, by_color=True):
    """8-connected objects that are several 4-connected objects, as (8-connected Obj, [4-connected Objs]); prints a summary."""
    four = objects(grid, 4, bg, by_color)
    owner = {cell: index for index, obj in enumerate(four) for cell in obj.cells}
    split = []
    for obj in objects(grid, 8, bg, by_color):
        parts = sorted({owner[cell] for cell in obj.cells})
        if len(parts) > 1:
            split.append((obj, [four[i] for i in parts]))
    print(f"{len(four)} objects under 4-connectivity, {len(four) - sum(len(p) - 1 for _, p in split)} under 8; "
          + (", ".join(f"color {o.color} at {o.bbox} splits into {len(p)}" for o, p in split) or "no difference"))
    return split


def fragments(grid, color, gap=1, bg=None):
    """Group the `color` objects (8-connected) whose cells come within `gap` empty cells of each other."""
    pieces = [obj for obj in objects(grid, 8, bg) if obj.color == color]
    parent = list(range(len(pieces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if any(max(abs(r - y), abs(c - x)) <= gap + 1 for r, c in pieces[i].cells for y, x in pieces[j].cells):
                parent[root(i)] = root(j)
    groups = {}
    for i, obj in enumerate(pieces):
        groups.setdefault(root(i), []).append(obj)
    return list(groups.values())


def bbox(grid_or_cells, bg=None):
    """Inclusive (r0, c0, r1, c1) of the given cells, or of a grid's non-background cells; None if there are none."""
    if grid_or_cells and isinstance(grid_or_cells[0], tuple):
        cells = grid_or_cells
    else:
        grid = grid_or_cells
        bg = background(grid) if bg is None else bg
        cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v != bg]
    if not cells:
        return None
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def crop(grid, box):
    r0, c0, r1, c1 = box
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]


def lines(grid, bg=None):
    """Full rows and columns of one non-background colour; cells where another full line crosses are allowed."""
    bg = background(grid) if bg is None else bg
    height, width = len(grid), len(grid[0])
    rows = {r: grid[r][0] for r in range(height) if grid[r][0] != bg and len(set(grid[r])) == 1}
    cols = {c: grid[0][c] for c in range(width) if grid[0][c] != bg and len({grid[r][c] for r in range(height)}) == 1}
    for _ in range(2):  # Allow crossings with the lines found so far.
        for r in range(height):
            values = {grid[r][c] for c in range(width) if c not in cols}
            if r not in rows and len(values) == 1 and values != {bg}:
                rows[r] = values.pop()
        for c in range(width):
            values = {grid[r][c] for r in range(height) if r not in rows}
            if c not in cols and len(values) == 1 and values != {bg}:
                cols[c] = values.pop()
    return {"rows": dict(sorted(rows.items())), "cols": dict(sorted(cols.items()))}


def split_by_lines(grid, bg=None):
    """Blocks between full lines: (line rows, line cols, blocks) where blocks[i][j] is the i-th band's j-th cell block."""
    found = lines(grid, bg)
    rows, cols = sorted(found["rows"]), sorted(found["cols"])

    def spans(cuts, size):
        edges, start = [], 0
        for cut in cuts + [size]:
            if cut > start:
                edges.append((start, cut - 1))
            start = cut + 1
        return edges

    blocks = [[crop(grid, (r0, c0, r1, c1)) for c0, c1 in spans(cols, len(grid[0]))]
              for r0, r1 in spans(rows, len(grid))]
    return rows, cols, blocks


def ray(grid, start, step, stop="non_bg", bg=None):
    """Cells visited from start (exclusive) moving by step until the edge or a stop cell (not included).

    stop: "non_bg" stops at any non-background cell, "edge" only at the grid edge, or a set of colours.
    Returns (cells, stop_cell, stop_color); stop_cell and stop_color are None when the ray left the grid.
    """
    bg = background(grid) if bg is None else bg
    (r, c), (dr, dc), cells = start, step, []
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return cells, None, None
        value = grid[r][c]
        if (stop == "non_bg" and value != bg) or (stop not in ("non_bg", "edge") and value in stop):
            return cells, (r, c), value
        cells.append((r, c))


def dihedral(grid):
    transpose = [list(row) for row in zip(*grid)]
    return {"identity": [list(row) for row in grid], "rot90": [row[::-1] for row in transpose],
            "rot180": [row[::-1] for row in grid[::-1]], "rot270": transpose[::-1],
            "flip_lr": [row[::-1] for row in grid], "flip_ud": [list(row) for row in grid[::-1]],
            "transpose": transpose, "anti_transpose": [row[::-1] for row in transpose[::-1]]}


def find_dihedral(a, b):
    return [name for name, grid in dihedral(a).items() if grid == b]


def _largest_empty_rectangle(grid, bg):
    """(height, width) of the largest all-background rectangle (histogram method)."""
    best, heights = (0, 0, 0), [0] * len(grid[0])
    for row in grid:
        heights = [h + 1 if v == bg else 0 for h, v in zip(heights, row)]
        stack = []
        for index, height in enumerate(heights + [0]):
            start = index
            while stack and stack[-1][1] >= height:
                start, top = stack.pop()
                best = max(best, (top * (index - start), top, index - start))
            stack.append((start, height))
    return best[1], best[2]


def _features(grid):
    bg = background(grid)
    counts = Counter(v for row in grid for v in row)
    fg = {color: n for color, n in counts.items() if color != bg}
    feats = {"H_in": len(grid), "W_in": len(grid[0]), "colors": len(counts), "fg_cells": sum(fg.values())}
    if fg:
        feats.update(min_color_count=min(fg.values()), max_color_count=max(fg.values()))
    for conn in (4, 8):
        found = objects(grid, conn, bg)
        feats[f"objects{conn}"] = len(found)
        if found:
            big = max(found, key=lambda o: o.size)
            feats[f"largest{conn}_h"] = big.bbox[2] - big.bbox[0] + 1
            feats[f"largest{conn}_w"] = big.bbox[3] - big.bbox[1] + 1
            feats[f"largest{conn}_size"] = big.size
    feats["objects_multicolor"] = len(objects(grid, 8, bg, by_color=False))
    box = bbox(grid, bg)
    if box:
        feats.update(content_h=box[2] - box[0] + 1, content_w=box[3] - box[1] + 1)
    found = lines(grid, bg)
    feats.update(line_rows=len(found["rows"]), line_cols=len(found["cols"]))
    feats["empty_rect_h"], feats["empty_rect_w"] = _largest_empty_rectangle(grid, bg)
    for color, n in counts.items():
        feats[f"count_{color}"] = n
    return feats


def shape_rules(puzzle=None, limit=8):
    """Rules out_dim = a*feature + b (a in -3..5, |b| <= 10) or a constant that fit every training pair.

    Prints, per output dimension, up to `limit` fitting rules (simplest first) with their test predictions, then the
    distinct predicted test shapes and how many rule pairs support each. Several fitting rules with different test
    predictions mean training does not settle the output size: choose with evidence, not by default.
    """
    puzzle = _puzzle(puzzle)
    train = [(_features(p["input"]), len(p["output"]), len(p["output"][0])) for p in puzzle["train"]]
    tests = [_features(t["input"]) for t in puzzle["test"]]
    names = sorted(set.intersection(*(set(f) for f, _, _ in train)) & set.intersection(*(set(f) for f in tests)))
    result = {}
    for dim, position in (("H_out", 1), ("W_out", 2)):
        targets = [row[position] for row in train]
        fits = []
        if len(set(targets)) == 1:
            fits.append((1, f"{dim} = {targets[0]} (constant)", [targets[0]] * len(tests)))
        for name in names:
            for a in (1, 2, 3, 4, 5, -1, -2, -3):
                b = targets[0] - a * train[0][0][name]
                if (abs(b) <= 10 and all(a * f[name] + b == t for (f, _, _), t in zip(train, targets))
                        and all(a * f[name] + b > 0 for f in tests)):
                    cost = (a != 1) + (a < 0) + (b != 0) + (abs(b) > 6) + (0 if name in ("H_in", "W_in") else 1)
                    text = f"{dim} = " + (f"{a}*" if a != 1 else "") + name + (f" {b:+d}" if b else "")
                    fits.append((cost, text, [a * f[name] + b for f in tests]))
        fits.sort(key=lambda item: (item[0], item[1]))
        result[dim] = [{"rule": text, "test": predicted} for _, text, predicted in fits]
        print(f"{dim}: {len(fits)} fitting rules" + "".join(f"\n  {text} -> test {predicted}"
                                                           for _, text, predicted in fits[:limit]))
    for index in range(len(tests)):
        support = Counter((h["test"][index], w["test"][index]) for h in result["H_out"] for w in result["W_out"])
        print(f"test {index} predicted shapes: " + (", ".join(f"{h}x{w} ({n})" for (h, w), n in support.most_common(5))
                                                    or "none fit"))
    return result
