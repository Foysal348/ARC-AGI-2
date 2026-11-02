"""E25 klee100 load test: the W4 teacher-forced reference dataset (foysalemonshanto/q27-e25-nb0-tf-ref).

Source: NB0's GPU output (esfersami50, notebook sha 37a427a2, 2026-10-31), downloaded to
D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb0-fp8-quality/nb0/ (tf_raw/*.npz + nb0_results.json). Only the arrays the
klee100 load notebook compares against are copied, unchanged (byte copies, sha256 recorded):
  A  = W4, BF16 KV, MTP off (seqs 8):  TF 2k / 8k / 40k / 90k fresh + repeat; the self-consistency set (decode cold / cached +
       the TF of prompt + decoded tokens) at 2k / 8k / 40k
  C  = W4, BF16 KV, MTP K3 (seqs 3):   TF 2k / 8k / 40k fresh + repeat (C 90k: the downloaded file is 0 bytes, D was not
       downloaded: both recorded as missing, never used)
nb0_reference.json: per file sha256 / bytes / ids sha (NB0's ids_sha formula) / prompt length; per boot and band NB0's own
recorded prompt sha256, variant, k and valid count (every array is checked against them here: the ids are byte-identical to
NB0's prompts); NB0's yardsticks (A/A, A~C, A~B = FP8 vs BF16 KV, C~D, prefix consistency, self-consistency).

Run: python build_e25_nb0_tf_ref_dataset.py  ->  D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref/ (the lead uploads it)
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

SRC = Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb0-fp8-quality/nb0")
NB0_NB = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/"
              r"Q27-XHIGH-E25_NB0_FP8_QUALITY.ipynb")
NB0_NB_SHA = "37a427a235bf665ee51c7a01767427a299720b4d80d6676f0ff2accb4510bf2c"      # the Kaggle-run NB0 (esfersami50)
OUT = Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref")
SLUG = "foysalemonshanto/q27-e25-nb0-tf-ref"
BANDS_A = ("2k", "8k", "40k", "90k")
BANDS_C = ("2k", "8k", "40k")
SC_BANDS = ("2k", "8k", "40k")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids):
    """NB0's ids_sha (nb0 CELL 6): sha256 of json.dumps([int(x) ...])."""
    return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()


def main():
    assert sha(NB0_NB.read_bytes()) == NB0_NB_SHA, "NB0 notebook is not the Kaggle-run version"
    res_raw = (SRC / "nb0_results.json").read_bytes()
    res = json.loads(res_raw)
    secs = res["sections"]["NB0"]
    boots = {"A": secs["A_nb0_A_bf16_mtp0"], "C": secs["C_nb0_C_bf16_mtp3"]}
    assert all(b["status"] == "DONE" for b in boots.values())
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    files, tf, sc = {}, {"A": {}, "C": {}}, {"A": {}}

    def take(name, want_sha=None, want_len=None, boot=None, band=None, run=None, kind=None):
        p = SRC / "tf_raw" / name
        b = p.read_bytes()
        assert len(b) > 0, f"{name}: empty file"
        z = np.load(p)
        ids = z["ids"]
        s = ids_sha(ids)
        if want_sha is not None:
            assert s == want_sha, f"{name}: ids sha {s} != NB0's recorded {want_sha}"
        if want_len is not None:
            assert len(ids) == want_len, f"{name}: {len(ids)} ids != {want_len}"
        (OUT / name).write_bytes(b)
        files[name] = {"bytes": len(b), "sha256": sha(b), "ids_sha256": s, "len": int(len(ids)), "keys": sorted(z.files),
                       "boot": boot, "band": band, "run": run, "kind": kind}
        return z

    for key, bands in (("A", BANDS_A), ("C", BANDS_C)):
        for band in bands:
            o = boots[key]["tf"][band]
            rec = {"variant": o["variant"], "prompt_sha256": o["prompt_sha256"], "prompt_len": o["prompt_len"], "aa": o.get("aa")}
            for run in ("fresh", "repeat"):
                m = o[run]
                assert m.get("error") is None and m.get("raw") and m["k"] == 5, (key, band, run, m)
                assert m["prompt_sha256"] == o["prompt_sha256"]
                take(m["raw"], m["prompt_sha256"], m["prompt_len"], key, band, run, "teacher_forced")
                rec[run] = {"file": m["raw"], "k": m["k"], "valid": m["valid"], "expected": m["expected"], "nonfinite": m["nonfinite"],
                            "cached_tokens": m["cached_tokens"], "note": m.get("note"), "sec": m.get("sec")}
            tf[key][band] = rec
    for band in SC_BANDS:
        o = boots["A"]["sc"][band]
        cold, cached, t = o["cold"], o["cached"], o["tf"]
        zc = take(cold["raw"], cold["prompt_sha256"], cold["prompt_len"], "A", band, "cold", "decode")
        take(cached["raw"], cached["prompt_sha256"], cached["prompt_len"], "A", band, "cached", "decode")
        zt = take(t["raw"], t["prompt_sha256"], t["prompt_len"], "A", band, "tf", "self_consistency_tf")
        P = cold["prompt_len"]
        assert np.array_equal(zt["ids"][:P], zc["ids"]) and np.array_equal(zt["ids"][P:], zc["tok"]), band
        sc["A"][band] = {"prompt_sha256": cold["prompt_sha256"], "prompt_len": P, "cold": cold["raw"], "cached": cached["raw"],
                         "tf": t["raw"], "tf_prompt_sha256": t["prompt_sha256"], "tokens": cold["tokens"],
                         "cached_tokens": [cold["cached_tokens"], cached["cached_tokens"]],
                         "decode_vs_tf": o.get("decode_vs_tf"), "cold_vs_cached": o.get("cold_vs_cached")}
    pc = res["prompt_check"]
    for band in BANDS_A:
        assert pc["tf_prompts"][band]["sha256"] == tf["A"][band]["prompt_sha256"], band
    for band in SC_BANDS:
        assert pc["sc_prompts"][band]["sha256"] == sc["A"][band]["prompt_sha256"], band
    an = res["analysis"]
    keep = ("status", "n", "expected", "mean_abs_dlogprob", "p99_abs_dlogprob", "max_abs_dlogprob", "rank1_agreement")
    yard = {}
    for pk in ("AB", "CD", "AC", "BD", "CE", "DE"):
        pr = an["pairs"][pk]
        yard[pk] = {"what": pr["what"], "ref": pr["ref"], "cand": pr["cand"],
                    "bands": {b: {"last256": {k: (e.get("last256") or {}).get(k) for k in keep},
                                  "full": {k: (e.get("full") or {}).get(k) for k in keep},
                                  "screening_gate": e.get("screening_gate"), "triage_gate": e.get("triage_gate")}
                              for b, e in pr["bands"].items()}}
    per = {k: {"aa": an["per_boot"][k]["aa"], "aa_full": an["per_boot"][k]["aa_full"],
               "prefix_consistency": an["per_boot"][k]["prefix_consistency"],
               "sc": {b: v.get("decode_vs_tf") for b, v in (an["per_boot"][k].get("sc") or {}).items()}} for k in ("A", "C", "B", "D")}
    ref = {"what": "E25 klee100 load test: NB0's W4 teacher-forced reference arrays (byte copies) + NB0's yardsticks",
           "source": {"nb0_notebook_sha256": NB0_NB_SHA, "nb0_kaggle": "esfersami50/q27-xhigh-e25-nb0-fp8-quality (2026-10-31, 64.1 min)",
                      "nb0_results_sha256": sha(res_raw), "local_copy": str(SRC)},
           "boots": {"A": "W4 (Merlin W4A16), BF16 KV, MTP off, seqs 8, FP8 capture list (captures to 32), prefix caching on",
                     "C": "W4 (Merlin W4A16), BF16 KV, MTP K3, seqs 3 (= RC2 BF16), prefix caching on"},
           "array_format": "teacher-forced npz: ids [n] int32 (the prompt), lp [n] float32 (logprob of ids[p] given ids[:p], p >= 1; "
                           "p = 0 is NaN), rank [n], top1 [n] (unique rank-1 id, -1 ambiguous / missing), top1_lp, topk_ids [n, 5], "
                           "topk_lp [n, 5], valid [n] bool; decode npz: ids (prompt), tok [256] greedy ignore_eos, lp, top1, top1_lp, "
                           "top2_lp (NB0 CELL 6 tf_request / decode)",
           "ids_sha_formula": "sha256(json.dumps([int(x) for x in ids]).encode()) (NB0 ids_sha)",
           "prompt_variants": {"tf": 5, "sc": {"2k": 6, "8k": 7, "40k": 6}},
           "files": files, "tf": tf, "sc": sc, "yardstick": {"pairs": yard, "per_boot": per, "rc2_anchor": an.get("rc2_anchor"),
                                                             "thresholds": an.get("thresholds")},
           "missing": {"C_90k": "the downloaded C_90k_fresh.npz is 0 bytes and C_90k_repeat.npz is absent (truncated Kaggle output "
                                "download); not used", "D_E_B": "not copied (not needed: the klee100 boots run BF16 KV only)"}}
    (OUT / "nb0_reference.json").write_text(json.dumps(ref, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    meta = {"title": "Q27 E25 NB0 TF Ref", "id": SLUG, "licenses": [{"name": "CC0-1.0"}],
            "description": "NB0 (E25, 2026-10-31) W4 BF16-KV teacher-forced raw arrays (A: MTP off, C: MTP K3) used as the reference "
                           "of the klee100 low-bit GPU load test. Byte copies; sha256 in nb0_reference.json."}
    (OUT / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    man = {n: v["sha256"] for n, v in sorted(files.items())}
    man["nb0_reference.json"] = sha((OUT / "nb0_reference.json").read_bytes())
    (OUT / "SHA256SUMS.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8", newline="\n")
    tot = sum(p.stat().st_size for p in OUT.iterdir())
    print("wrote", OUT, "|", len(list(OUT.iterdir())), "files |", tot, "bytes | nb0_reference.json", man["nb0_reference.json"])
    return man


if __name__ == "__main__":
    main()
