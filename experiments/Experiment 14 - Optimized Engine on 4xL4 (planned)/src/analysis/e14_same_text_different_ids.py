import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures/tok_int4")
cases = ["\n\n", "\n\n\n", "    return x\n", " grid", "def f(grid):", "</think>\n\n", "1, 2, 3"]
for s in cases:
    canon = tok(s, add_special_tokens=False)["input_ids"]
    # a non-canonical split: tokenize each character separately and concatenate
    alt = [i for ch in s for i in tok(ch, add_special_tokens=False)["input_ids"]]
    same_text = tok.decode(alt) == tok.decode(canon) == s
    print(repr(s), "canon", canon, "alt", alt, "same_text", same_text, "ids_differ", alt != canon,
          "re-encode(alt)==canon", tok(tok.decode(alt), add_special_tokens=False)["input_ids"] == canon)
