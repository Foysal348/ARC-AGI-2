

import json
import numpy as np
from transformers import AutoTokenizer


# ── Grid helpers ──────────────────────────────────────────────────────────────

def grid_to_str(grid) -> str:
    """Convert a 2-D integer grid to a compact newline-separated string."""
    return "\n".join("".join(str(int(c)) for c in row) for row in grid)

# Legacy alias kept for compatibility
convert_grid_to_string = grid_to_str


def is_valid_grid(guess) -> bool:
    """Return True if *guess* is a 2-D numpy array with shape within 1–30."""
    return (
        isinstance(guess, np.ndarray)
        and guess.ndim == 2
        and all(1 <= x <= 30 for x in guess.shape)
    )

# Legacy alias
is_valid_solution = is_valid_grid


def shuffled(lst):
    return np.random.permutation(lst).tolist()


# ── Colour-permutation helpers ────────────────────────────────────────────────

def permute_mod(a, descriptor, invert: bool = False):
    """Apply (or invert) a colour permutation encoded in *descriptor*."""
    perm = [int(ch) for ch in descriptor if ch.isdigit()]
    assert sorted(perm) == list(range(10)), "Descriptor must encode a full permutation of 0-9"
    a = np.asarray(a)
    if a.ndim == 3:
        # One-hot / channel axis: invert by argsort
        if not invert:
            perm = np.argsort(perm)
        a = a[..., perm]
    else:
        assert a.ndim == 2
        if invert:
            perm = np.argsort(perm)
        a = np.asarray(perm)[a]
    return a


def permute_rnd_all_(query):
    """Return a random colour-permutation descriptor string."""
    perm = np.random.permutation(10).tolist()
    return "permute" + "".join(map(str, perm))


# ── Formatter ─────────────────────────────────────────────────────────────────

class QwenFormatter:
    """
    Converts ARC tasks to/from the Qwen chat-template token format used during
    fine-tuning and inference.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    # ── Formatting ────────────────────────────────────────────────────────────

    def fmt_query(self, query) -> str:
        return "<|im_start|>user\n" + grid_to_str(query[0]["input"]) + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return grid_to_str(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge: bool = False) -> str:
        if last_is_challenge:
            test  = train[-1]
            train = train[:-1]
        else:
            test = None

        parts = []
        for ex in train:
            parts.append(
                "<|im_start|>user\n"
                + grid_to_str(ex["input"])
                + "<|im_end|><|im_start|>assistant\n"
                + grid_to_str(ex["output"])
                + "<|im_end|>"
            )
        text = "".join(parts)

        if test is not None:
            text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
        return text

    # ── Token-count helpers ───────────────────────────────────────────────────

    def max_new_tokens(self) -> int:
        """Upper bound on output tokens for a 30×30 grid."""
        dummy = np.zeros([30, 30], dtype=int)
        return len(self.tokenizer.encode(self.fmt_reply([dummy]))) + 1

    # ── Decoding ──────────────────────────────────────────────────────────────

    def tokens_to_grid(self, tokens, limit_rows: int = 30):
        """Try to parse a token sequence back to a 2-D numpy array."""
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines   = text.strip().split("\n")
            rows    = [
                [int(ch) for ch in line if ch.isdigit()]
                for line in lines
            ]
            rows    = [r for r in rows if r][:limit_rows]
            arr     = np.array(rows, dtype=int)
            if is_valid_grid(arr):
                return arr
        except Exception:
            pass
        return None

    # Backwards-compatible alias
    def convert_tokens_to_array(self, tokens, limit_rows=30):
        return self.tokens_to_grid(tokens, limit_rows)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ArcDataset:
    """
    Wraps the raw ARC JSON data and exposes augmentation, formatting and
    submission helpers.
    """

    # ── Transformation helpers ────────────────────────────────────────────────

    @staticmethod
    def forward_mod(a, key, use_perm: bool = True):
        if a is None:
            return a
        for op in key.split(".")[1:]:
            if   op == "rot90":              a = np.rot90(a)
            elif op == "transpose":          a = np.swapaxes(a, 0, 1)
            elif op.startswith("permute"):   a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith(("copy", "out", "ex", "run")):
                pass  # identity ops
            else:
                raise NotImplementedError(f"Unknown forward op: '{op}'")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm: bool = True):
        if a is None:
            return a
        for op in key.split(".")[1:][::-1]:
            if   op == "rot90":              a = np.rot90(a, k=3)
            elif op == "transpose":          a = np.swapaxes(a, 0, 1)
            elif op.startswith("permute"):   a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith(("copy", "out", "ex", "run")):
                pass
            else:
                raise NotImplementedError(f"Unknown invert op: '{op}'")
        return a

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, queries, replies=None, keys=None, is_orig: bool = False):
        if replies is None:
            replies = {}
        if keys is not None:
            keys = [k for k in keys if k is not None]
        self.queries  = queries if keys is None else {k: queries[k] for k in keys}
        self.replies  = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig  = is_orig
        self.keys     = sorted(queries.keys()) if keys is None else list(keys)
        self._transposed = None   # lazy cache

    def change_keys(self, keys, keep_flags: bool = False):
        flags = {"is_orig": self.is_orig} if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, path, keys=None):
        with open(path) as fh:
            raw = fh.read()
        return cls(queries=json.loads(raw), is_orig=True, keys=keys)

    def load_replies(self, path):
        print(f"*** Loading solutions from '{path}' ...")
        with open(path) as fh:
            parsed = json.loads(fh.read())
        self.replies = {k: parsed[k] for k in self.keys}
        return self

    # ── Splitting / merging ───────────────────────────────────────────────────

    def split_multi_replies(self):
        """Expand tasks with multiple test inputs into separate entries."""
        idx = [(k, i) for k in self.keys for i in range(len(self.queries[k]["test"]))]
        return self.__class__(
            keys    = [f"{k}_{i}" for k, i in idx],
            queries = {
                f"{k}_{i}": {"train": self.queries[k]["train"], "test": [self.queries[k]["test"][i]]}
                for k, i in idx
            },
            replies = {
                f"{k}_{i}": [self.replies[k][i]]
                for k, i in idx if k in self.replies
            },
        )

    def shuffled(self):
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    @staticmethod
    def append(*datasets):
        return datasets[0].__class__(
            queries = {k: v for d in datasets for k, v in d.queries.items()},
            replies = {k: v for d in datasets for k, v in d.replies.items()},
            keys    = [k    for d in datasets for k    in d.keys           ],
        )

    # ── Augmentation primitives ───────────────────────────────────────────────

    def mod_single(self, mod_func, descriptor, i, keep_key: bool, inputs_only: bool):
        queries, replies, keys = {}, {}, []
        for k0 in self.keys:
            if descriptor is None:
                desc = ("copy{i}" if mod_func is np.copy else mod_func.__name__).format(i=i)
            elif isinstance(descriptor, str):
                desc = descriptor.format(i=i)
            else:
                desc = descriptor(self.queries[k0]).format(i=i)

            def apply(a, d=desc):
                return np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()

            prefix = "I" if inputs_only else ""
            k1 = k0 if keep_key else f"{k0}.{prefix}{desc}"
            keys.append(k1)
            queries[k1] = {
                split: [
                    {t: (apply(v) if (t == "input" or not inputs_only) else v) for t, v in pair.items()}
                    for pair in pairs
                ]
                for split, pairs in self.queries[k0].items()
            }
            if k0 in self.replies:
                replies[k1] = [apply(a) for a in self.replies[k0]]
        return self.__class__(queries=queries, replies=replies, keys=keys)

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False,
            keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        stack = mod_func.__name__.startswith("rot") if stack is None else stack
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        for i in range(n):
            cur = (cur if stack else self).mod_single(
                mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only
            )
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    # ── Example-order shuffling ───────────────────────────────────────────────

    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys, new_queries, new_replies = [], {}, {}
        for key in self.keys:
            n = len(self.queries[key]["train"])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None:
                p = p[:keep_max]
            sep  = "-" if p.max() > 9 else ""
            nk   = f"{key}.ex{sep.join(map(str, p.tolist()))}"
            new_keys.append(nk)
            new_queries[nk] = {
                split: (np.array(v, dtype=object)[p].tolist() if split == "train" else v)
                for split, v in self.queries[key].items()
            }
            if key in self.replies:
                new_replies[nk] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    # ── High-level augmentation ───────────────────────────────────────────────

    def augment(self, n: int = 1, shfl_keys: bool = False, seed: int = 42):
        """
        Full augmentation pipeline:
          1. identity + transpose  (×2)
          2. × rot90 stack         (×4 rotations, kept)
          3. × n random colour-permutations
          4. shuffle example order
        """
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose,  keep=True)
        d = d.mod(np.rot90,      n=3, keep=True)
        d = d.mod(permute_mod,   permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d

    # ── Token-length management ───────────────────────────────────────────────

    def get_length(self, key, formatter, name, max_of_transposed=False):
        if formatter is None:
            if name == "input":
                return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name == "reply":
                return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else:
                raise ValueError(name)
        datasets = [self]
        if max_of_transposed:
            if self._transposed is None:
                self._transposed = self.mod(np.transpose, keep=False, keep_key=True)
            datasets.append(self._transposed)
        return max(
            len(formatter.tokenizer.encode(ds.get(key, formatter=formatter)[name]))
            for ds in datasets
        )

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        tmp = self.change_keys(self.keys)
        new_keys, new_queries, new_replies = [], {}, {}
        for key in self.keys:
            reply = tmp.replies.get(key)
            while max_len < tmp.get_length(key, formatter=formatter, name=name):
                query = tmp.queries[key]
                if not key.split(".")[-1].startswith("ex"):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                parts = key.split(".")
                assert parts[-1].startswith("ex")
                trimmed = parts[-1][2:-1] if from_end else parts[-1][3:]
                key = ".".join(parts[:-1] + [f"ex{trimmed}"])
                tmp.queries[key] = {
                    split: (v[:-1] if from_end else v[1:]) if split == "train" else v
                    for split, v in query.items()
                }
                if reply is not None:
                    tmp.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = tmp.queries[key]
            if reply is not None:
                new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    # ── Formatting ────────────────────────────────────────────────────────────

    def get(self, key, formatter: QwenFormatter) -> dict:
        train = formatter.fmt_train(self.queries[key]["train"])
        query = formatter.fmt_query(self.queries[key]["test"])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ""
        text  = (train + query + reply) if reply else formatter.fmt_train(
            self.queries[key]["train"], last_is_challenge=True
        )
        return dict(key=key, train=train, query=query, reply=reply, input=train + query, text=text)

    def as_list(self, formatter: QwenFormatter):
        return [self.get(k, formatter) for k in self.keys]

    # ── Submission helpers ────────────────────────────────────────────────────

    def get_submission(self, results=None) -> dict:
        assert self.is_orig, "Must be called on the original (is_orig=True) dataset."
        sub = {
            k: [{f"attempt_{i+1}": [[0]] for i in range(2)}
                for _ in range(len(self.queries[k]["test"]))]
            for k in self.keys
        }
        if results is not None:
            self.fill_submission(results, sub)
        return sub

    @staticmethod
    def fill_submission(results, submission):
        print(f"*** Writing {len(results)} outputs into submission ...")
        for k, v in results.items():
            base_id, base_nr = k.rsplit("_", 1)
            target = submission[base_id][int(base_nr)]
            for i, grid in enumerate(v[: len(target)]):
                target[f"attempt_{i+1}"] = grid.tolist()

    def validate_submission(self, submission) -> float:
        assert self.is_orig, "Must be called on the original dataset."
        score = 0.0
        for k, outputs in self.replies.items():
            for idx, ref in enumerate(outputs):
                for attempt in ("attempt_1", "attempt_2"):
                    if np.array_equal(ref, submission[k][idx][attempt]):
                        score += 1.0 / len(outputs)
                        break
        return score
