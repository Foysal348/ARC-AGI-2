"""checks: (1) get_puzzle_priority in puzzle_order.py is byte-identical to both reference starter.py files;
(2) order_task_ids == the reference's own sorted(...) call; (3) prints tiers and the EVAL12 positions.
usage: python test_puzzle_order.py <arc-agi_evaluation_challenges.json>"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import puzzle_order as po

src = (HERE / "puzzle_order.py").read_text(encoding="utf-8")
cut = lambda s: s[s.index("def get_puzzle_priority"):s.index("def ", s.index("def get_puzzle_priority") + 5)]
for ref in ("ref_v1_starter.py.txt", "ref_v2_starter.py.txt"):
    r = (HERE / ref).read_text(encoding="utf-8")
    assert cut(src).rstrip() == cut(r).rstrip(), f"get_puzzle_priority differs from {ref}"
print("get_puzzle_priority: byte-identical to both references")

data = json.load(open(sys.argv[1]))
ref_order = sorted(data.keys(), key=lambda k: po.get_puzzle_priority(k, data))   # the reference's exact call
assert po.order_task_ids(data) == ref_order
units = po.order_units(data)
assert len(units) == sum(len(t["test"]) for t in data.values())
tiers = [po.get_puzzle_priority(k, data)[0] for k in ref_order]
print(f"{len(data)} tasks / {len(units)} units | Easy {tiers.count(1)} Hard {tiers.count(2)} Extreme {tiers.count(3)}")
print("first 5:", ref_order[:5], "| last 5:", ref_order[-5:])
EVAL12 = ["136b0064", "20a9e565", "21897d95", "247ef758", "2b83f449", "4c3d4a41", "4c7dc4dd", "6e4f6532",
          "88bcf3b4", "898e7135", "8b7bacbf", "b6f77b65"]
pos = {k: i for i, k in enumerate(ref_order)}
for k in sorted((k for k in EVAL12 if k in data), key=pos.get):
    t, f = po.get_puzzle_priority(k, data)
    print(f"  EVAL12 {k}: rank {pos[k] + 1}/{len(data)} tier {t} fine {f:.4f} tests {len(data[k]['test'])}")
print("PUZZLE ORDER TEST PASS")
