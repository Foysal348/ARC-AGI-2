import ast,hashlib,json,difflib
from pathlib import Path
import nbformat
D=Path(__file__).resolve().parent;ROOT=D.parent
SRC=D/'sources' if (D/'sources/Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb').exists() else ROOT
base=nbformat.read(SRC/'Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb',as_version=4)
r3=nbformat.read((SRC/'Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R3_REPORT_ONLY.ipynb') if SRC!=ROOT else (ROOT/'Q27_E8_EXACT_ARCH_R3_REPORT_ONLY/Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R3_REPORT_ONLY.ipynb'),as_version=4)
r5=nbformat.read(D/'Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb',as_version=4)
donor_loader=(SRC/'_donor_codex_cell3.txt').read_text().replace('%%writefile arc_loader.py\n','')
assert len(r5.cells)==12
assert all(r5.cells[i].source==base.cells[i].source for i in range(7))
# the donor vendored utility is intentionally only annotation+unused import adapted.
A=ast.parse(donor_loader);B=ast.parse(r5.cells[8].source)
functions=['permute_mod','permute_rnd_all_']
methods=['augment','mod','mod_single','shuffle_ex','forward_mod','invert_mod']
cd=next(n for n in A.body if isinstance(n,ast.ClassDef) and n.name=='ArcDataset')
cn=next(n for n in B.body if isinstance(n,ast.ClassDef) and n.name=='ArcDataset')
for name in methods:
    x=next(n for n in cd.body if isinstance(n,ast.FunctionDef) and n.name==name)
    y=next(n for n in cn.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False),name
for name in functions:
    x=next(n for n in A.body if isinstance(n,ast.FunctionDef) and n.name==name)
    y=next(n for n in B.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False),name
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={
 'baseline_deadlines_equal':r5.cells[2].source==base.cells[2].source,
 'base_source_cells_0_through_6_equal':all(r5.cells[i].source==base.cells[i].source for i in range(7)),
 'base_scheduler_modified_for_request_routing':r5.cells[10].source!=base.cells[7].source,
 'donor_AST_exact_core_ops':methods+functions,
 'donor_QwenFormatter':'unused tokenizer type annotation removed',
 'base_original_path':'Q27-XHIGH-v1_4_4_E7_ORDERING_ONLY.ipynb',
 'donor_original_path':'arc-2026d-with-codex (24)(1).ipynb',
 'r3_original_path':'Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R3_REPORT_ONLY.ipynb',
 'r5_sha256':sha(D/'Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb'),
 'gpu_validated':False,
}
(D/'SOURCE_AUDIT.json').write_text(json.dumps(report,indent=2))
(D/'SOURCE_COMPARISON.md').write_text('''# R5 source audit and donor parity

## E7 baseline -> R5
- E7 cells 0–6 (**initial markdown and cells 0–5**) source identical.
- E7 configuration, all admission/drain/hard-stop timestamps, primary model/sampling/49,152 +8,192 token allowances, ordering, seeding, MTP/FP8 settings **unchanged**.
- E7 scheduler cell 6 is intentionally patched: after `finish` it checkpoints full thinking, queues candidate scoring; output dispatch differentiates `e8|` score requests from `uid|phase` primary requests; both consume global concurrency. The scheduler is therefore **not byte-identical** and primary GPU throughput can still change.
- E7 final report is adapted to compute baseline E7 vs E8 final pass@2; mixed throughput projection is flagged invalid.

## Codex donor -> R5
- Exact AST match for donor `ArcDataset.augment`, `mod`, `mod_single`, `shuffle_ex`, `forward_mod`, `invert_mod`, `permute_mod`, `permute_rnd_all_`.
- `QwenFormatter` only type annotation/unused `AutoTokenizer` import adaptation.
- **Not copied:** donor 4B per-task LoRA, Torch `past_key_values` recursive `turbo_dfs`, donor beam/probmul metadata; those are not directly compatible with Q27's vLLM engine.
- Donor 8 views retained; Q27 scoring uses `prompt_logprobs` on the same vLLM engine, with one scoring request in flight.

## R4 -> R5
- Text-offset chronology across fenced, JSON and row-run candidates; conservative context filtering of scratch/example/incorrect grids.
- Original fallback is protected when identity/mirror operation is supported by every training pair; unknown hidden correctness is still unproven.
- Primary unit admission is processed before scorer admission at each scheduling step (deadlines unchanged; shared GPU interference remains).
- Score-token decode + joint boundary checks fail closed on unverifiable formats; this can reduce scoring coverage on real tokenizer.
- Pending cancellation counts unique active/current/queued units once.
- Auxiliary E8 report is written after EVAL final metrics; embedded E8 report refreshed.
- 0/1/2+ routing and donor eight-view architecture retained; no new model or speculative DFS added.

## Not validated
- Real 4xL4 vLLM0.19 / FP8-KV / MTP3 prompt-logprob API memory and throughput.
- Whether mined grids include any **additional correct** answer on real Q27 runs (full reasoning wasn't saved in prior artifacts).
- Primary pass@1/coverage under mixed GPU contention and pass@2 gain; require paired GPU tests.
- Normal Kaggle completion was not run; only CPU and fake-engine tests.
''')
print(json.dumps(report,indent=2))
