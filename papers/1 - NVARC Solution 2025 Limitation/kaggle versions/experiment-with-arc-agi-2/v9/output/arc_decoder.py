import os
import bz2
import pickle
import numpy as np


def hashable(guess):
    return tuple(map(tuple, guess))


def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs


def getter_full_probmul_3(guesses, baseline=None):
    # Dynamic baseline to avoid negative scores (from V1)
    if baseline is None:
        beam_scores = [g["beam_score"] for g in guesses]
        if not beam_scores:
            return 0
        baseline = max(beam_scores) + 1.0
    inf_score = np.sum([baseline - g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score


def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)


def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score


def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


SELECTOR_CURRENT = "CURRENT"
SELECTOR_IMBUE_EXACT = "IMBUE_EXACT"
SELECTOR_IMBUE_QWEN_DIVERSE = "IMBUE_QWEN_DIVERSE"


class ArcDecoder:

    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    @staticmethod
    def _validate_decoded_sample(sample):
        """Validate and normalize one persisted decoder candidate.

        Required fields:
          - solution: non-empty 2-D ARC grid, integer colors 0..9, max 30x30
          - beam_score: finite scalar
          - score_aug: non-empty 1-D sequence of finite scalars

        Returns:
            (normalized_sample, error_message)
        """
        if not isinstance(sample, dict):
            return None, f"payload is {type(sample).__name__}, expected dict"

        required = ("solution", "beam_score", "score_aug")
        missing = [name for name in required if name not in sample]
        if missing:
            return None, f"missing required field(s): {missing}"

        try:
            solution = np.asarray(sample["solution"])
        except Exception as exc:
            return None, f"solution cannot be converted to ndarray: {exc}"

        if solution.ndim != 2:
            return None, f"solution ndim={solution.ndim}, expected 2"
        if solution.size == 0 or any(int(d) <= 0 for d in solution.shape):
            return None, f"solution has empty/invalid shape {solution.shape}"
        if any(int(d) > 30 for d in solution.shape):
            return None, f"solution shape {solution.shape} exceeds ARC 30x30 bounds"

        try:
            solution_float = solution.astype(np.float64)
        except Exception as exc:
            return None, f"solution is not numeric: {exc}"

        if not np.all(np.isfinite(solution_float)):
            return None, "solution contains NaN/Inf"
        if not np.all(solution_float == np.floor(solution_float)):
            return None, "solution contains non-integer values"
        if np.any(solution_float < 0) or np.any(solution_float > 9):
            return None, "solution contains colors outside 0..9"
        solution = solution_float.astype(np.int64)

        try:
            beam_score = float(sample["beam_score"])
        except Exception as exc:
            return None, f"beam_score is not a scalar number: {exc}"
        if not np.isfinite(beam_score):
            return None, "beam_score is NaN/Inf"

        try:
            score_aug = np.asarray(sample["score_aug"], dtype=np.float64)
        except Exception as exc:
            return None, f"score_aug is not numeric: {exc}"
        if score_aug.ndim != 1:
            return None, f"score_aug ndim={score_aug.ndim}, expected 1"
        if score_aug.size == 0:
            return None, "score_aug is empty"
        if not np.all(np.isfinite(score_aug)):
            return None, "score_aug contains NaN/Inf"

        normalized = dict(sample)
        normalized["solution"] = solution
        normalized["beam_score"] = beam_score
        normalized["score_aug"] = score_aug.tolist()
        return normalized, None

    def load_decoded_results(self, store, run_name=""):
        """Load only well-formed outputs belonging to this exact split dataset.

        Stale/foreign files, corrupt payloads, and malformed samples are skipped
        with explicit warnings so they cannot crash or contaminate selectors.
        """
        self.decoded_results = {}
        expected_keys = set(getattr(self.dataset, "keys", []))
        skipped_foreign = []
        skipped_bad = []

        if not os.path.isdir(store):
            print(f"WARNING: decoded-results directory does not exist: {store}")
            return

        for key in sorted(os.listdir(store)):
            path = os.path.join(store, key)
            if not os.path.isfile(path):
                continue

            base_key = key.split(".")[0]
            if base_key not in expected_keys:
                skipped_foreign.append(key)
                continue

            try:
                with bz2.BZ2File(path) as f:
                    outputs = pickle.load(f)
            except Exception as exc:
                skipped_bad.append((key, f"read error: {exc}"))
                continue

            if not isinstance(outputs, (list, tuple)):
                skipped_bad.append((key, f"unexpected payload type: {type(outputs).__name__}"))
                continue

            valid_outputs = []
            for i, sample in enumerate(outputs):
                normalized, error = self._validate_decoded_sample(sample)
                if error is not None:
                    skipped_bad.append((f"{key}.out{i}", error))
                    continue
                valid_outputs.append((i, normalized))

            if not valid_outputs:
                continue

            self.decoded_results.setdefault(base_key, {})
            for i, sample in valid_outputs:
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

        loaded_keys = set(self.decoded_results)
        missing_expected = sorted(expected_keys - loaded_keys)
        print(
            f"*** Decoded coverage: {len(loaded_keys)}/{len(expected_keys)} expected test outputs "
            f"({len(missing_expected)} missing)"
        )
        if missing_expected:
            print(f"WARNING: missing decoded outputs (first 20): {missing_expected[:20]}")
        if skipped_foreign:
            print(f"WARNING: skipped {len(skipped_foreign)} stale/foreign output files: {skipped_foreign[:10]}")
        if skipped_bad:
            print(f"WARNING: skipped {len(skipped_bad)} invalid output entries: {skipped_bad[:10]}")

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    @staticmethod
    def _first_distinct(candidates, reference):
        """Return the first grid whose exact grid hash differs from reference."""
        if reference is None:
            return candidates[0] if candidates else None
        reference_hash = hashable(reference)
        for cand in candidates:
            if hashable(cand) != reference_hash:
                return cand
        return None

    def run_current_selection(self):
        """Exact compatibility copy of the pre-Imbue notebook's run_dual_selection().

        IMPORTANT: score_sum() already deduplicates grids. Therefore, whenever
        score_full_probmul_3 returns >=2 grids, the first two entries of this
        pool are simply Probmul rank-1 and rank-2; Kgmon cannot affect attempt-2.
        We preserve this method unchanged as the CURRENT control arm.
        """
        results = {}
        for bk, guesses in self.decoded_results.items():
            g = {k: v for k, v in guesses.items()}
            ranked_a = score_full_probmul_3(g)
            ranked_b = score_kgmon(g)

            candidates_pool = []
            seen = set()
            for cand in ranked_a[:3] + ranked_b[:3]:
                h = hashable(cand)
                if h not in seen:
                    seen.add(h)
                    candidates_pool.append(cand)

            top_a = candidates_pool[0] if candidates_pool else None
            top_b = candidates_pool[1] if len(candidates_pool) > 1 else top_a
            results[bk] = [x for x in [top_a, top_b] if x is not None]
        return results

    def run_imbue_exact_selection(self):
        """Portable literal analogue of Imbue's ARC attempt-2 rule.

        Imbue sorts its population by one global fitness score and, independently
        for every challenge/test input, chooses the highest-ranked output that is
        different from attempt-1. In this notebook each test input is already a
        separate base_key (task_id_testIndex), so the closest portable analogue is:
          attempt_1 = Probmul rank-1
          attempt_2 = highest Probmul-ranked exact-grid disagreement.

        Since score_sum() already deduplicates outputs, this normally equals
        Probmul rank-2. This arm exists to verify that fact empirically rather
        than assuming it.
        """
        results = {}
        for bk, guesses in self.decoded_results.items():
            ranked = score_full_probmul_3({k: v for k, v in guesses.items()})
            if not ranked:
                results[bk] = []
                continue
            attempt_1 = ranked[0]
            attempt_2 = self._first_distinct(ranked[1:], attempt_1)
            if attempt_2 is None:
                attempt_2 = attempt_1
            results[bk] = [attempt_1, attempt_2]
        return results

    def run_imbue_qwen_diverse_selection(self):
        """Imbue-inspired heterogeneous-ranker attempt-2 for this Qwen/NVARC decoder.

        Keep the established Probmul rank-1 as the conservative attempt-1.
        For attempt-2, scan the independent Kgmon ranking and take its highest
        exact-grid disagreement with attempt-1. Fall back to the Probmul ranking
        only when Kgmon provides no distinct output.

        This is NOT claimed to be Imbue's literal selector; it ports Imbue's
        highest-quality-disagreement principle to the two existing local rankers.
        """
        results = {}
        for bk, guesses in self.decoded_results.items():
            g = {k: v for k, v in guesses.items()}
            ranked_a = score_full_probmul_3(g)
            ranked_b = score_kgmon(g)
            if not ranked_a:
                results[bk] = []
                continue

            attempt_1 = ranked_a[0]
            attempt_2 = self._first_distinct(ranked_b, attempt_1)
            if attempt_2 is None:
                attempt_2 = self._first_distinct(ranked_a[1:], attempt_1)
            if attempt_2 is None:
                attempt_2 = attempt_1

            results[bk] = [attempt_1, attempt_2]
        return results

    # Backward-compatible name. CURRENT stays the safe control until the A/B
    # benchmark below demonstrates a gain for another selector.
    def run_dual_selection(self):
        return self.run_current_selection()

    @staticmethod
    def available_selectors():
        return (SELECTOR_CURRENT, SELECTOR_IMBUE_EXACT, SELECTOR_IMBUE_QWEN_DIVERSE)

    def run_selector(self, selector_name=SELECTOR_CURRENT):
        name = str(selector_name).strip().upper() or SELECTOR_CURRENT
        methods = {
            SELECTOR_CURRENT: self.run_current_selection,
            SELECTOR_IMBUE_EXACT: self.run_imbue_exact_selection,
            SELECTOR_IMBUE_QWEN_DIVERSE: self.run_imbue_qwen_diverse_selection,
        }
        if name not in methods:
            raise ValueError(f"Unknown selector '{selector_name}'. Choose one of {tuple(methods)}")
        return methods[name]()

    def benchmark_selection_algos(self, expected_keys=None):
        """Benchmark rankers/selectors on *all* expected labelled Eval outputs.

        Missing decoded outputs count as misses.  Foreign decoded keys are ignored.
        This prevents an incomplete inference run from inflating pass@2 accuracy.
        """
        print("*** Benchmark selection algorithms + pass@2 selectors...")

        all_labelled_keys = [
            k for k in getattr(self.dataset, "keys", [])
            if k in getattr(self.dataset, "replies", {})
        ]

        if expected_keys is None:
            expected_keys = all_labelled_keys
            scope_label = "ALL LABELLED EVAL"
        else:
            requested = list(dict.fromkeys(expected_keys))
            labelled_set = set(all_labelled_keys)
            unknown = [k for k in requested if k not in labelled_set]
            if unknown:
                print(f" WARNING: benchmark scope ignored {len(unknown)} unknown/unlabelled keys: {unknown[:10]}")
            expected_keys = [k for k in requested if k in labelled_set]
            scope_label = "EXPLICIT DEBUG/RUN SCOPE"

        if not expected_keys:
            raise ValueError("benchmark_selection_algos has no labelled expected keys to score")

        print(f" benchmark scope: {scope_label} ({len(expected_keys)} expected outputs)")

        labels = {k: self.dataset.replies[k][0] for k in expected_keys}
        expected_set = set(expected_keys)

        num_tasks_per_puzzle = {}
        for basekey in expected_keys:
            puzzle_id, _ = basekey.rsplit("_", 1)
            num_tasks_per_puzzle[puzzle_id] = num_tasks_per_puzzle.get(puzzle_id, 0) + 1

        num_puzzles = len(num_tasks_per_puzzle)
        decoded_expected = expected_set & set(self.decoded_results)
        missing_expected = sorted(expected_set - decoded_expected)
        foreign_decoded = sorted(set(self.decoded_results) - expected_set)

        print(
            f" expected labelled outputs: {len(expected_keys)} | "
            f"decoded: {len(decoded_expected)} | missing: {len(missing_expected)} | "
            f"puzzles: {num_puzzles}"
        )
        if missing_expected:
            print(f" missing outputs count as misses (first 20): {missing_expected[:20]}")
        if foreign_decoded:
            print(f" WARNING: ignoring {len(foreign_decoded)} foreign decoded keys: {foreign_decoded[:10]}")

        num_solved_candidates = 0
        num_total_candidates = 0
        correct_beam_scores = []

        # Candidate-level diagnostics only over available expected outputs.
        for basekey in expected_keys:
            correct_solution = labels[basekey]
            basevalues = self.decoded_results.get(basekey, {})
            for subkey, sample in basevalues.items():
                solution = sample["solution"]
                beam_score = sample.get("beam_score", np.nan)
                score_aug = sample.get("score_aug", [])
                aug_mean = np.mean(score_aug) if len(score_aug) else np.nan

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_candidates += 1
                    if np.isfinite(beam_score):
                        correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"
                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_candidates += 1

        print(f" candidate hits: {num_solved_candidates}/{num_total_candidates}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")
        else:
            print(" avg correct beam score: N/A (no correct solutions found)")
            print(" max correct beam score: N/A (no correct solutions found)")

        def score_selected(selected):
            # Iterate over ALL expected labels, not only decoded_results.
            correct_keys = set()
            for k, correct_solution in labels.items():
                attempts = selected.get(k, [])[:self.n_guesses]
                if any(np.array_equal(guess, correct_solution) for guess in attempts):
                    correct_keys.add(k)

            score = sum(
                1.0 / num_tasks_per_puzzle[k.rsplit("_", 1)[0]]
                for k in correct_keys
            )
            return score, correct_keys

        # Original single-ranker top-2 baselines.
        ranker_metrics = {}
        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            score, correct_keys = score_selected(selected)
            ranker_metrics[name] = {"score": score, "correct_keys": correct_keys}
            print(f" acc: {score:5.2f}/{num_puzzles:3} ('{name}') hits={len(correct_keys)} outputs")

        # Selector A/B/C benchmark. CURRENT and IMBUE_EXACT are kept as a
        # control/sanity pair even though they are structurally equivalent here.
        selector_metrics = {}
        selected_cache = {}
        print("\n*** Selector A/B/C benchmark (two attempts each) ***")
        for name in self.available_selectors():
            selected = self.run_selector(name)
            selected_cache[name] = selected
            score, correct_keys = score_selected(selected)
            selector_metrics[name] = {"score": score, "correct_keys": correct_keys}
            print(f" {name:24s}: {score:5.2f}/{num_puzzles:3}  hits={len(correct_keys)} outputs")

        current_score = selector_metrics[SELECTOR_CURRENT]["score"]
        for name in (SELECTOR_IMBUE_EXACT, SELECTOR_IMBUE_QWEN_DIVERSE):
            delta = selector_metrics[name]["score"] - current_score
            print(f" delta {name:19s} vs CURRENT: {delta:+.4f}")

        current_selected = selected_cache[SELECTOR_CURRENT]
        exact_selected = selected_cache[SELECTOR_IMBUE_EXACT]
        comparable_keys = sorted(expected_set)
        current_equals_exact = all(
            len(current_selected.get(k, [])) == len(exact_selected.get(k, []))
            and all(
                np.array_equal(a, b)
                for a, b in zip(current_selected.get(k, []), exact_selected.get(k, []))
            )
            for k in comparable_keys
        )
        print(f" CURRENT == IMBUE_EXACT on this candidate pool: {current_equals_exact}")

        best_selector = max(
            self.available_selectors(),
            key=lambda name: selector_metrics[name]["score"],
        )
        print(f" best selector on this labelled run: {best_selector}")
        print(" NOTE: production selector is NOT auto-switched; keep CURRENT until repeated validation supports a change.")

        return {
            "num_puzzles": num_puzzles,
            "num_expected_outputs": len(expected_keys),
            "num_decoded_outputs": len(decoded_expected),
            "missing_outputs": missing_expected,
            "foreign_decoded_keys": foreign_decoded,
            "rankers": ranker_metrics,
            "selectors": selector_metrics,
            "current_equals_imbue_exact": current_equals_exact,
            "best_selector": best_selector,
        }
