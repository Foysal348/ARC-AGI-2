
"""
arc_decoder.py
==============
Post-inference scoring/ranking: aggregates beam hypotheses from multiple
augmented views and ranks them using two complementary scoring strategies.
"""

import os
import bz2
import pickle
import numpy as np


# ── Utilities ─────────────────────────────────────────────────────────────────

def to_hashable(grid):
    return tuple(map(tuple, grid))


def rank_by_score(guesses: dict, score_fn) -> list:
    """
    Deduplicate hypotheses by grid content, compute aggregate score per unique
    solution, and return solutions ordered best-first.
    """
    buckets = {}
    for g in guesses.values():
        h = to_hashable(g["solution"])
        bucket = buckets.setdefault(h, ([], g["solution"]))
        bucket[0].append(g)

    ranked = sorted(
        ((score_fn(members), sol) for members, sol in buckets.values()),
        key=lambda x: x[0],
        reverse=True,
    )
    return [sol for _, sol in ranked]


# ── Scoring functions ─────────────────────────────────────────────────────────

def _score_probmul(guesses, baseline: float = 3.0) -> float:
    """
    Combined beam + augmented-view NLL score.
    Higher → better candidate.
    """
    beam_part = float(np.sum([baseline - g["beam_score"] for g in guesses]))
    aug_part  = float(np.mean([
        np.sum([baseline - s for s in g["score_aug"]]) for g in guesses
    ]))
    return beam_part + aug_part


def _score_vote(guesses) -> float:
    """
    Voting + mean augmented NLL.
    More votes and lower augmented NLL → higher score.
    """
    vote_count = len(guesses)
    aug_mean   = float(np.mean([np.mean(g["score_aug"]) for g in guesses]))
    return vote_count - aug_mean


def score_probmul(guesses: dict) -> list:
    return rank_by_score(guesses, _score_probmul)

# Legacy alias
score_full_probmul_3 = score_probmul


def score_vote(guesses: dict) -> list:
    return rank_by_score(guesses, _score_vote)

# Legacy alias
score_kgmon = score_vote


# Combined ensemble: re-rank using both signals then interleave
def score_ensemble(guesses: dict) -> list:
    """
    Interleave results from both ranking strategies.
    This exposes more diversity in attempt_1 / attempt_2.
    """
    ranked_prob = score_probmul(guesses)
    ranked_vote = score_vote(guesses)
    seen, result = set(), []
    for a, b in zip(ranked_prob, ranked_vote):
        for cand in (a, b):
            h = to_hashable(cand)
            if h not in seen:
                seen.add(h)
                result.append(cand)
    return result


SELECTION_ALGORITHMS = [score_probmul, score_vote, score_ensemble]


# ── ArcDecoder ────────────────────────────────────────────────────────────────

class ArcDecoder:
    """
    Loads per-puzzle beam outputs saved by the worker and selects the best
    two guesses for submission.
    """

    def __init__(self, dataset, n_guesses: int = 2):
        self.dataset         = dataset
        self.n_guesses       = n_guesses
        self.decoded_results = {}   # base_key → {sub_key → sample_dict}

    def load_decoded_results(self, store: str, run_tag: str = ""):
        """Load all .bz2 pickle files from *store*."""
        for fname in os.listdir(store):
            path     = os.path.join(store, fname)
            base_key = fname.split(".")[0]
            self.decoded_results.setdefault(base_key, {})
            with bz2.BZ2File(path) as fh:
                outputs = pickle.load(fh)
            for idx, sample in enumerate(outputs):
                tag = f"{fname}{run_tag}.out{idx}"
                self.decoded_results[base_key][tag] = sample

    def run_selection(self, algorithm=score_vote) -> dict:
        return {
            bk: algorithm({k: g for k, g in v.items()})
            for bk, v in self.decoded_results.items()
        }

    # Backwards-compatible alias
    def run_selection_algo(self, selection_algorithm=score_vote):
        return self.run_selection(selection_algorithm)

    def benchmark_selection_algos(self):
        print("=" * 60)
        print("  Selection-algorithm benchmark")
        print("=" * 60)

        labels              = {}
        tasks_per_puzzle    = {}
        n_correct_keys      = 0
        n_total_keys        = 0
        correct_beam_scores = []

        for base_key, base_vals in self.decoded_results.items():
            puzzle_id, sub_idx = base_key.split("_")
            tasks_per_puzzle[puzzle_id] = max(
                tasks_per_puzzle.get(puzzle_id, 0), int(sub_idx) + 1
            )
            ref = self.dataset.replies[base_key][0]
            labels[base_key] = ref

            for sub_key, sample in base_vals.items():
                sol    = sample["solution"]
                bscore = sample["beam_score"]
                amean  = float(np.mean(sample["score_aug"]))

                if np.shape(ref) != np.shape(sol):
                    tag = "bad_shape"
                elif np.array_equal(ref, sol):
                    tag = "CORRECT"
                    n_correct_keys += 1
                    correct_beam_scores.append(bscore)
                else:
                    tag = "wrong"

                if tag == "CORRECT":
                    sz = f"{sol.shape[0]}x{sol.shape[1]}"
                    print(f"  {tag} beam={bscore:7.4f} aug={amean:7.4f} {sz:5s} [{sub_key}]")
                n_total_keys += 1

        print(f"\nSub-key recall : {n_correct_keys}/{n_total_keys}")
        if correct_beam_scores:
            print(f"Beam score (correct) — mean={np.mean(correct_beam_scores):.4f}  max={np.max(correct_beam_scores):.4f}")

        n_puzzles = len(tasks_per_puzzle)
        print()
        for algo in SELECTION_ALGORITHMS:
            name     = algo.__name__
            selected = self.run_selection(algo)
            solved   = {
                k for k, cands in selected.items()
                if any(np.array_equal(c, labels[k]) for c in cands[: self.n_guesses])
            }
            score = sum(1.0 / tasks_per_puzzle[k.split("_")[0]] for k in solved)
            print(f"  [{name:20s}]  {score:5.1f} / {n_puzzles}  ({100*score/max(n_puzzles,1):.1f}%)")
