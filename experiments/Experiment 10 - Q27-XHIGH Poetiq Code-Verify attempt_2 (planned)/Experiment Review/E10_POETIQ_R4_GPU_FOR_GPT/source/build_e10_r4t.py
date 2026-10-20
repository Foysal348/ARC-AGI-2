"""E10 Poetiq R4T = R4 + full thinking saved per iteration (Claude, 2026-10-14, user request). Deterministic.

Source: Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4.ipynb (sha 596bf694...). Logging only - generation, prompts,
verification, voting and scoring are unchanged.
Change (cell 7, e10_after_decode): every finished iteration writes the complete generated text (thinking + forced-close
stop text + answer/code, special tokens kept so </think> stays visible) to
  q27x/poetiq_thinking/<task>_e<expert>_i<iteration>.txt
plus a small header (task, expert, iteration, mode, finish reason, prompt/p1/p2 tokens). A write failure is logged
and never stops the solver. Size: ~0.3 MB per iteration.
usage: python build_e10_r4t.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4.ipynb"
SRC_SHA = "596bf694ce65a4d2360051b5c065d996df4c9851e5d09ebb0bcad03c3d597cce"
OUT = HERE.parent / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4T.ipynb"


def once(text, old, new, where):
    if text.count(old) != 1:
        sys.exit(f"replacement not unique ({text.count(old)}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


raw = SRC.read_bytes()
assert hashlib.sha256(raw).hexdigest() == SRC_SHA, "R4 source changed"
nb = json.loads(raw)
c = nb["cells"]
src = lambda i: "".join(c[i]["source"])

t = src(7)
t = once(t, "E10_PROGRAM_DIR = OUT / 'poetiq_programs'; E10_PROGRAM_DIR.mkdir(parents=True,exist_ok=True)\n",
         "E10_PROGRAM_DIR = OUT / 'poetiq_programs'; E10_PROGRAM_DIR.mkdir(parents=True,exist_ok=True)\n"
         "E10_THINKING_DIR = OUT / 'poetiq_thinking'; E10_THINKING_DIR.mkdir(parents=True,exist_ok=True)   # R4T: full thinking per iteration\n",
         "thinking dir")
t = once(t, "    text, closed=e10_answer_text(req)\n    code=poetiq_parse_code(text)\n",
         "    text, closed=e10_answer_text(req)\n"
         "    code=poetiq_parse_code(text)\n"
         "    try:   # R4T: save the complete generation (thinking + stop text + answer); logging only\n"
         "        _full=tokenizer.decode(list(req['p1_ids'])+list(req['p2_prefix'])+list(req['p2_ids']),skip_special_tokens=False)\n"
         "        _head=(f\"# task={st['task_id']} expert={st['expert']} iteration={st['iteration']+1} mode={req['mode'] or 'natural'} \"\n"
         "               f\"finish={reason} prompt_tokens={len(req['prompt_ids'])} p1_tokens={len(req['p1_ids'])} p2_tokens={len(req['p2_ids'])}\\n\")\n"
         "        (E10_THINKING_DIR/f\"{st['task_id']}_e{st['expert']}_i{st['iteration']+1}.txt\").write_text(_head+_full,encoding='utf-8')\n"
         "    except Exception as _exc:\n"
         "        print('[E10] thinking save failed:',st['task_id'],repr(_exc)[:200],flush=True)\n",
         "save thinking")
c[7]["source"] = t.splitlines(keepends=True)
c[0]["source"] = (src(0) + "\n\n**R4T (2026-10-14):** R4 + the full thinking of every iteration saved to `q27x/poetiq_thinking/` "
                  "(logging only).\n").splitlines(keepends=True)
for cell in c:
    if cell["cell_type"] == "code":
        cell["outputs"] = []; cell["execution_count"] = None
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
OUT.write_bytes(data)
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())
