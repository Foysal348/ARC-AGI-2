"""CPU tests for e14_nccl_bench and e14_trace_summary (no GPU, no torch; Windows or Linux).

Run from this directory:
  python -m pytest -q -p no:cacheprovider --noconftest --rootdir=. --confcutdir=. --import-mode=importlib \
      --basetemp=<new dir> test_e14_nccl_bench_and_trace.py
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import e14_nccl_bench as B  # noqa: E402
import e14_trace_summary as T  # noqa: E402

BENCH = str(HERE / "e14_nccl_bench.py")
SUMMARY = str(HERE / "e14_trace_summary.py")
PY = sys.executable


def bench_args(tmp_path, **kw):
    """Driver args as the CLI would build them."""
    argv = ["--out", str(tmp_path / "result.json")]
    a = B.build_parser().parse_args(argv)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ============================================================================================ bench: module + fake
def test_modules_import_without_torch():
    code = ("import sys; sys.path.insert(0, %r); import e14_nccl_bench, e14_trace_summary; "
            "print('torch' in sys.modules)" % str(HERE))
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "False"


def test_fake_run_full_structure(tmp_path):
    out = tmp_path / "result.json"
    p = subprocess.run([PY, BENCH, "--fake", "--out", str(out)], capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["fake"] is True and r["torch_imported_in_driver"] is False
    for k in ("topology", "arms", "summary", "config", "started", "finished", "bench_version"):
        assert k in r
    topo = r["topology"]
    assert topo["fake"] is True and topo["p2p_all"] is True
    assert "GPU0" in topo["nvidia_smi_topo_m"]["out"]
    assert len(topo["p2p_matrix"]) == 4
    assert set(r["arms"]) == {"baseline", "p2p_sys", "shm_memcpy"}
    for name, a in r["arms"].items():
        assert a["status"] == "ok", (name, a.get("error"))
        assert a["env"]["NCCL_DEBUG"] == "INFO"
        assert a["env"]["NCCL_DEBUG_SUBSYS"] == "INIT,GRAPH,ENV"
        assert a["env"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "8"
        assert a["env"]["NCCL_CUMEM_ENABLE"].startswith("0") or "(inherited)" in a["env"]["NCCL_CUMEM_ENABLE"]
        assert a["correct"] is True
        assert a["ranks_reported"] == [0, 1, 2, 3]
        assert a["transport"] and a["transport"][0]["count"] == 8      # 4 ranks x 2 channels
        assert a["nccl"]["nccl_version"] == "2.29.7+cuda13.0"
        assert a["nccl"]["init_complete"] == 4
        assert set(a["eager"]) == {l for l, _ in B.EAGER_SIZES}
        for label, nb in B.EAGER_SIZES:
            e = a["eager"][label]
            assert e["bytes"] == nb and len(e["per_rank"]) == 4
            for k in ("median_us", "p90_us", "algbw_GBps", "busbw_GBps", "loop_us_per_call", "per_iter_max_median_us"):
                assert e[k] is not None
            assert e["busbw_GBps"] == pytest.approx(e["algbw_GBps"] * 1.5)
        g = a["graph_983040"]
        assert g["calls_per_graph"] == 140 and g["replays"] == 10 and g["graph_correct"] is True
        base = B._FAKE_BASE_US[name] * 1.09                            # slowest fake rank (3) has jitter 1.09
        assert base <= g["per_call_us"] <= base * 1.02 + 1e-9
        assert g["replay_ms_median"] == pytest.approx(g["per_call_us"] * 140 / 1000)
        ag = a["allgather_11_9MB"]
        assert ag["bytes_per_rank"] == B.AG_BYTES_PER_RANK and ag["total_bytes"] == 4 * B.AG_BYTES_PER_RANK
        assert ag["eager"]["median_ms"] > 0 and ag["graph"]["status"] == "ok"
        assert "fake child" in a["stderr_tail"]
    assert r["arms"]["p2p_sys"]["env"]["NCCL_P2P_LEVEL"] == "SYS"
    assert r["arms"]["shm_memcpy"]["env"]["NCCL_SHM_USE_CUDA_MEMCPY"] == "1"
    assert r["arms"]["p2p_sys"]["primary_transport"] == "P2P/IPC/read"
    assert r["arms"]["baseline"]["primary_transport"] == "SHM/direct/direct"
    s = r["summary"]
    assert s["fake"] is True and s["best_arm"] == "p2p_sys"
    assert s["improvement_vs_baseline_pct"] == pytest.approx(
        (1 - s["best_per_call_us"] / s["baseline_per_call_us"]) * 100)
    assert s["adopt_candidate"] == "p2p_sys"
    assert set(s["per_arm"]) == set(r["arms"])
    assert s["per_arm"]["baseline"]["per_step_comm_ms_est"] == pytest.approx(
        140 * s["per_arm"]["baseline"]["graph_per_call_us"] / 1000 + s["per_arm"]["baseline"]["allgather_graph_ms"])


def test_fake_no_p2p_skips_p2p_arm(tmp_path):
    out = tmp_path / "r.json"
    p = subprocess.run([PY, BENCH, "--fake", "--fake-p2p", "no", "--out", str(out), "--arms",
                        "baseline,p2p_sys"], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["topology"]["p2p_all"] is False
    a = r["arms"]["p2p_sys"]
    assert a["status"] == "skipped" and a["reason"].startswith("skipped: no P2P")
    assert a["env"]["NCCL_P2P_LEVEL"] == "SYS"
    assert r["arms"]["baseline"]["status"] == "ok"
    assert r["summary"]["best_arm"] == "baseline" and r["summary"]["adopt_candidate"] is None


def test_force_p2p_runs_p2p_arm(tmp_path):
    a = bench_args(tmp_path, fake=True, fake_p2p="no", arms="p2p_sys", force_p2p=True)
    r = B.driver_main(a, [])
    assert r["arms"]["p2p_sys"]["status"] == "ok"


def test_driver_never_raises_bad_child_and_unknown_arm(tmp_path):
    a = bench_args(tmp_path, fake=True, arms="baseline,nonexistent,shm_memcpy")
    r = B.driver_main(a, [], cmd_overrides={"baseline": [PY, "-c", "import sys; sys.exit(3)"]})
    on_disk = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert on_disk["arms"].keys() == r["arms"].keys()
    assert r["arms"]["baseline"]["status"] == "error"
    assert "exit code 3" in r["arms"]["baseline"]["error"]
    assert r["arms"]["nonexistent"]["status"] == "error" and "unknown arm" in r["arms"]["nonexistent"]["error"]
    assert r["arms"]["shm_memcpy"]["status"] == "ok"
    assert r["summary"]["best_arm"] == "shm_memcpy"
    assert r["summary"]["baseline_per_call_us"] is None and r["summary"]["adopt_candidate"] is None


def test_driver_survives_probe_crash_and_interrupt(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(B, "probe_topology", boom)
    a = bench_args(tmp_path, fake=True, arms="baseline,p2p_sys")
    r = B.driver_main(a, [])
    assert "probe exploded" in r["topology"]["error"]
    assert r["arms"]["baseline"]["status"] == "ok"
    assert r["arms"]["p2p_sys"]["status"] == "skipped"          # no topology -> no P2P claim

    def interrupt(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(B, "run_one_arm", interrupt)
    out2 = tmp_path / "r2.json"
    a2 = bench_args(tmp_path, fake=True, arms="baseline", out=str(out2))
    r2 = B.driver_main(a2, [])
    assert "KeyboardInterrupt" in r2["driver_error"]
    assert json.loads(out2.read_text(encoding="utf-8"))["finished"]


def test_extra_arm_env_spec(tmp_path):
    t = B.parse_arm_env_specs(["ll:NCCL_PROTO=LL,NCCL_ALGO=Ring", "ch2:NCCL_MIN_NCHANNELS=2"])
    assert t == {"ll": {"NCCL_PROTO": "LL", "NCCL_ALGO": "Ring"}, "ch2": {"NCCL_MIN_NCHANNELS": "2"}}
    a = bench_args(tmp_path, fake=True, arms="ll", arm_env=["ll:NCCL_PROTO=LL"])
    r = B.driver_main(a, [])
    assert r["arms"]["ll"]["status"] == "ok" and r["arms"]["ll"]["env"]["NCCL_PROTO"] == "LL"


def test_build_arm_env_respects_inherited_and_no_sglang_env(tmp_path):
    a = bench_args(tmp_path)
    env, applied = B.build_arm_env({"NCCL_P2P_LEVEL": "SYS"}, a, base_env={"NCCL_CUMEM_ENABLE": "1", "PATH": "x",
                                                                           "NCCL_IB_DISABLE": "1"})
    assert env["NCCL_CUMEM_ENABLE"] == "1" and applied["NCCL_CUMEM_ENABLE"] == "1 (inherited)"
    assert env["NCCL_NVLS_ENABLE"] == "0" and env["NCCL_GRAPH_REGISTER"] == "0"
    assert env["NCCL_P2P_LEVEL"] == "SYS" and env["NCCL_DEBUG"] == "INFO"
    assert applied["NCCL_IB_DISABLE"] == "1 (inherited)"
    a.no_sglang_env = True
    a.nccl_debug_subsys = "INIT,P2P,SHM"
    env2, applied2 = B.build_arm_env({}, a, base_env={"PATH": "x"})
    assert "NCCL_CUMEM_ENABLE" not in env2 and "CUDA_DEVICE_MAX_CONNECTIONS" not in env2
    assert env2["NCCL_DEBUG_SUBSYS"] == "INIT,P2P,SHM"


# ============================================================================================ bench: timeouts
def test_run_with_timeout_kills_sleeper(tmp_path):
    t0 = time.monotonic()
    r = B.run_with_timeout([PY, "-c", "import time; time.sleep(60)"], dict(os.environ), 1.5,
                           str(tmp_path / "o.log"), str(tmp_path / "e.log"))
    assert r["status"] == "timeout"
    assert time.monotonic() - t0 < 30
    assert r["elapsed_s"] >= 1.4


def test_run_with_timeout_kills_grandchild(tmp_path):
    marker, started = tmp_path / "alive.txt", tmp_path / "started.txt"
    gc = tmp_path / "grandchild.py"
    gc.write_text("import sys, time\ntime.sleep(4)\nopen(sys.argv[1], 'w').write('alive')\n", encoding="utf-8")
    ch = tmp_path / "child.py"
    ch.write_text("import subprocess, sys, time\n"
                  "subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])\n"
                  "open(sys.argv[3], 'w').write('ok')\n"
                  "time.sleep(60)\n", encoding="utf-8")
    t0 = time.monotonic()
    r = B.run_with_timeout([PY, str(ch), str(gc), str(marker), str(started)], dict(os.environ), 2.5,
                           str(tmp_path / "o.log"), str(tmp_path / "e.log"))
    assert r["status"] == "timeout"
    assert started.exists(), "child did not start its grandchild before the timeout"
    while time.monotonic() - t0 < 7.0:
        time.sleep(0.25)
    assert not marker.exists(), "grandchild survived the process-group kill"


def test_run_with_timeout_ok_and_error(tmp_path):
    ok = B.run_with_timeout([PY, "-c", "print('hi')"], dict(os.environ), 30, str(tmp_path / "o"), str(tmp_path / "e"))
    assert ok["status"] == "ok" and ok["returncode"] == 0
    assert (tmp_path / "o").read_text().strip() == "hi"
    bad = B.run_with_timeout([str(tmp_path / "no_such_binary_xyz")], dict(os.environ), 5, str(tmp_path / "o2"),
                             str(tmp_path / "e2"))
    assert bad["status"] == "error" and "spawn failed" in bad["error"]


def test_arm_timeout_keeps_partial_rank_results(tmp_path):
    stub = tmp_path / "stub_child.py"
    stub.write_text(
        "import json, time\n"
        "json.dump({'rank': 0, 'world': 4, 'errors': {}, 'phases_done': ['init', 'correct'],\n"
        "           'correct': {'pass': True, 'expected': 10.0, 'min': 10.0, 'max': 10.0}}, open('rank_0.json', 'w'))\n"
        "print('host:1:1 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct', flush=True)\n"
        "time.sleep(60)\n", encoding="utf-8")
    a = bench_args(tmp_path, fake=True, arms="baseline", timeout_per_arm=2.0)
    t0 = time.monotonic()
    r = B.driver_main(a, [], cmd_overrides={"baseline": [PY, str(stub)]})
    assert time.monotonic() - t0 < 60
    arm = r["arms"]["baseline"]
    assert arm["status"] == "timeout" and "killed" in arm["error"]
    assert arm["ranks_reported"] == [0]
    assert arm["correct"] is False                                   # 1 of 4 ranks is not a pass
    assert arm["primary_transport"] == "SHM/direct/direct"
    assert arm["phases_done"] == {"0": ["init", "correct"]}


# ============================================================================================ bench: NCCL log parsing
SHM_LOG = """\
[rank0]:[W1023 10:00:01.000000000 ProcessGroupNCCL.cpp:4561] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 ...
3a1f2c0e9b7d:812:812 [0] NCCL INFO Bootstrap: Using eth0:172.19.2.2<0>
3a1f2c0e9b7d:812:812 [0] NCCL INFO cudaDriverVersion 13000
3a1f2c0e9b7d:812:812 [0] NCCL INFO NCCL version 2.29.7+cuda13.0
3a1f2c0e9b7d:812:901 [0] NCCL INFO NET/Plugin: Could not find: libnccl-net.so.
3a1f2c0e9b7d:812:901 [0] NCCL INFO NCCL_CUMEM_ENABLE set by environment to 0.
3a1f2c0e9b7d:812:901 [0] NCCL INFO NCCL_GRAPH_REGISTER set by environment to 0.
3a1f2c0e9b7d:812:901 [0] NCCL INFO ncclCommInitRankConfig comm 0x55d0c8a0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 1000 commId 0x9b2c4f1e - Init START
3a1f2c0e9b7d:812:901 [0] NCCL INFO comm 0x55d0c8a0 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
3a1f2c0e9b7d:812:901 [0] NCCL INFO Pattern 4, crossNic 0, nChannels 2, bw 6.000000/6.000000, type SYS/PIX, sameChannels 1
3a1f2c0e9b7d:812:901 [0] NCCL INFO  0 : GPU/0-1000 (0/5000/-1) GPU/1-2000 (1/5000/-1)
3a1f2c0e9b7d:812:901 [0] NCCL INFO Channel 00/02 : 0 1 2 3\r
3a1f2c0e9b7d:812:901 [0] NCCL INFO Channel 01/02 : 0 1 2 3
3a1f2c0e9b7d:812:901 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
3a1f2c0e9b7d:812:901 [0] NCCL INFO P2P Chunksize set to 131072
3a1f2c0e9b7d:812:901 [0] NCCL INFO Check P2P Type isAllDirectP2p 0 directMode 0
3a1f2c0e9b7d:812:930 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct
3a1f2c0e9b7d:812:930 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via SHM/direct/direct
3a1f2c0e9b7d:813:931 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via SHM/direct/direct
3a1f2c0e9b7d:813:931 [1] NCCL INFO Channel 01/0 : 1[1] -> 2[2] via SHM/direct/direct
3a1f2c0e9b7d:814:932 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via SHM/direct/direct
3a1f2c0e9b7d:814:932 [2] NCCL INFO Channel 01/0 : 2[2] -> 3[3] via SHM/direct/direct
3a1f2c0e9b7d:815:933 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via SHM/direct/direct
3a1f2c0e9b7d:815:933 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via SHM/direct/direct
3a1f2c0e9b7d:815:933 [3] NCCL INFO Channel 01/0 : 3[3] -> 0[0] via SHM/direct/direct
3a1f2c0e9b7d:812:930 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 0
3a1f2c0e9b7d:812:930 [0] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via SHM/direct/direct
3a1f2c0e9b7d:812:930 [0] NCCL INFO Connected all trees
3a1f2c0e9b7d:812:901 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
3a1f2c0e9b7d:812:901 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
3a1f2c0e9b7d:812:901 [0] NCCL INFO CC Off, workFifoBytes 1048576
3a1f2c0e9b7d:812:901 [0] NCCL INFO ncclCommInitRankConfig comm 0x55d0c8a0 rank 0 nranks 4 cudaDev 0 nvmlDev 0 busId 1000 commId 0x9b2c4f1e - Init COMPLETE
3a1f2c0e9b7d:812:901 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 4 total 0.61 (kernels 0.21, alloc 0.02, bootstrap 0.00, allgathers 0.00, topo 0.19, graphs 0.00, connections 0.14, rest 0.05)
3a1f2c0e9b7d:813:901 [1] NCCL INFO ncclCommInitRankConfig comm 0x55d0c8b0 rank 1 nranks 4 cudaDev 1 nvmlDev 1 busId 2000 commId 0x9b2c4f1e - Init COMPLETE
"""

CUMEM_LOG = """\
gpu-node:4410:4502 [0] NCCL INFO NCCL version 2.27.5+cuda12.9
gpu-node:4410:4502 [0] NCCL INFO NCCL_P2P_LEVEL set by environment to SYS
gpu-node:4410:4502 [0] NCCL INFO Check P2P Type isAllDirectP2p 1 directMode 0
gpu-node:4410:4511 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM/read
gpu-node:4410:4511 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM/read
gpu-node:4411:4512 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via P2P/CUMEM/read
gpu-node:4412:4513 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/CUMEM/read
gpu-node:4413:4514 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via P2P/CUMEM/read
gpu-node:4413:4514 [3] NCCL INFO Channel 00/0 : 3[3] -> 2[2] via P2P/CUMEM/CE
gpu-node:4410:4502 [0] NCCL INFO Pattern 4, crossNic 0, nChannels 2, bw 12.000000/12.000000, type PHB/PIX, sameChannels 1
"""

IPC_NET_LOG = """\
host-a:77:88 [2] NCCL INFO Channel 00/0 : 2[2] -> 3[3] via P2P/IPC/read
host-a:77:88 [2] NCCL INFO Channel 01/0 : 2[2] -> 3[3] via P2P/IPC/read
host-a:77:88 [2] NCCL INFO Channel 02/0 : 2[2] -> 3[3] via P2P/IPC/read
host-a:77:88 [2] NCCL INFO Channel 00/0 : 2[2] -> 1[1] via P2P/IPC
host-a:77:88 [3] NCCL INFO Channel 00/0 : 3[3] -> 4[0] [send] via NET/Socket/0
host-b:91:99 [0] NCCL INFO Channel 00/0 : 3[3] -> 4[0] [receive] via NET/Socket/0
host-a:77:88 [2] NCCL WARN Cuda failure 'peer access is not supported between these two devices'
host-a:77:88 [2] NCCL INFO P2P is disabled between connected GPUs 2 and 3
"""


def test_parse_shm_log():
    p = B.parse_nccl_log(SHM_LOG)
    assert p["nccl_version"] == "2.29.7+cuda13.0" and p["cuda_driver"] == 13000
    assert p["env"] == {"NCCL_CUMEM_ENABLE": "0", "NCCL_GRAPH_REGISTER": "0"}
    assert p["primary"] == "SHM/direct/direct" and p["families"] == ["SHM"]
    assert len(p["connections"]) == 9                       # the duplicate 3->0 ch01 line is deduplicated
    row = p["transport"][0]
    assert row["count"] == 9 and row["channels"] == [0, 1]
    assert row["pairs"] == ["0->1", "1->0", "1->2", "2->3", "3->0"]
    assert p["rings"] == [{"channel": 0, "order": [0, 1, 2, 3]}, {"channel": 1, "order": [0, 1, 2, 3]}]
    assert p["patterns"][0]["nChannels"] == 2 and p["patterns"][0]["type_intra"] == "SYS"
    assert p["coll_channels"][0]["coll"] == 2 and p["coll_channels"][0]["p2p"] == 2
    assert p["p2p_type"] == ["isAllDirectP2p 0 directMode 0"]
    assert p["init_complete"] == 2 and p["init_total_s"] == [0.61]
    assert "all rings, use ring PXN 0 GDR 0" in p["connected"]
    assert p["warnings"] == []


def test_parse_env_value_strips_trailing_period():
    p = B.parse_nccl_log("h:1:1 [0] NCCL INFO NCCL_CUMEM_ENABLE set by environment to 0.\n"
                         "h:1:1 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0\n")
    assert p["env"] == {"NCCL_CUMEM_ENABLE": "0", "NCCL_SOCKET_IFNAME": "eth0"}


def test_parse_cumem_log():
    p = B.parse_nccl_log(CUMEM_LOG)
    assert p["nccl_version"] == "2.27.5+cuda12.9"
    assert p["env"]["NCCL_P2P_LEVEL"] == "SYS"
    assert p["primary"] == "P2P/CUMEM/read"
    assert p["families"] == ["P2P"]
    assert [t["transport"] for t in p["transport"]] == ["P2P/CUMEM/read", "P2P/CUMEM/CE"]
    assert p["transport"][0]["count"] == 5
    assert p["p2p_type"] == ["isAllDirectP2p 1 directMode 0"]


def test_parse_ipc_and_net_log():
    p = B.parse_nccl_log(IPC_NET_LOG)
    tr = {t["transport"]: t for t in p["transport"]}
    assert tr["P2P/IPC/read"]["count"] == 3 and tr["P2P/IPC"]["count"] == 1
    assert tr["P2P/IPC/read"]["channels"] == [0, 1, 2]
    assert tr["NET/Socket/0"]["count"] == 2                  # [send] and [receive] are distinct
    assert {c["dir"] for c in p["connections"] if c["family"] == "NET"} == {"send", "receive"}
    assert p["families"] == ["NET", "P2P"]
    assert p["primary"] == "P2P/IPC/read"
    assert any("peer access" in w for w in p["warnings"])
    assert any("P2P is disabled" in n for n in p["p2p_notes"])


def test_parse_empty_and_garbage():
    assert B.parse_nccl_log("")["primary"] is None
    p = B.parse_nccl_log("random text\nTraceback (most recent call last):\nNCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct\n")
    assert p["primary"] == "SHM/direct/direct"                # prefix-less line still parsed


# ============================================================================================ bench: math + aggregation
def test_bandwidth_math():
    alg = B.algbw_GBps(983040, 100.0)
    assert alg == pytest.approx(9.8304)
    assert B.busbw_allreduce(alg, 4) == pytest.approx(14.7456)
    alg64 = B.algbw_GBps(64 * 2 ** 20, 20900.0)               # the measured 64 MiB eager all-reduce
    assert alg64 == pytest.approx(3.2110, abs=1e-4)
    assert B.busbw_allreduce(alg64, 4) == pytest.approx(4.8165, abs=1e-4)
    assert B.busbw_allgather(10.0, 4) == pytest.approx(7.5)
    assert B.busbw_allreduce(10.0, 2) == pytest.approx(10.0)
    assert B.algbw_GBps(1, 0) is None and B.busbw_allreduce(None, 4) is None


def test_percentile_and_stats():
    assert B.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == pytest.approx(9.1)
    assert B.percentile([5], 90) == 5 and B.percentile([], 50) is None
    s = B.stats_us([3, 1, 2])
    assert s["median_us"] == 2 and s["min_us"] == 1 and s["max_us"] == 3 and s["n"] == 3


def _rank(r, mult, ok=True):
    return {"rank": r, "world": 4, "errors": {}, "phases_done": ["init"],
            "correct": {"pass": ok, "expected": 10.0},
            "eager": {"983040B": {"samples_us": [100 * mult, 110 * mult, 120 * mult], "loop_us_per_call": 90 * mult}},
            "graph_983040": {"calls": 140, "replay_us": [140 * 200 * mult, 140 * 210 * mult, 140 * 220 * mult],
                             "eager_loop_us_per_call": 250 * mult, "graph_correct": True,
                             "capture_mode": "thread_local"},
            "allgather_eager": {"bytes_per_rank": 1000, "samples_us": [1000 * mult] * 3}}


def test_aggregate_max_over_ranks():
    ranks = [_rank(r, 1 + 0.1 * r) for r in range(4)]
    a = B.aggregate_ranks(ranks, 4)
    assert a["correct"] is True
    e = a["eager"]["983040B"]
    assert e["median_us"] == pytest.approx(110 * 1.3)            # slowest rank's median
    assert e["min_rank_median_us"] == pytest.approx(110)
    assert e["p90_us"] == pytest.approx(B.percentile([130, 143, 156], 90))
    assert e["per_iter_max_median_us"] == pytest.approx(143)
    assert e["slowest_rank"] == "3" and e["loop_us_per_call"] == pytest.approx(117)
    assert e["per_rank"]["0"]["median_us"] == pytest.approx(110)
    assert e["busbw_GBps"] == pytest.approx(B.algbw_GBps(983040, 143) * 1.5)
    g = a["graph_983040"]
    assert g["per_call_us"] == pytest.approx(210 * 1.3) and g["min_rank_per_call_us"] == pytest.approx(210)
    assert g["graph_correct"] is True and g["eager_loop_per_call_us"] == pytest.approx(325)
    ag = a["allgather_11_9MB"]
    assert ag["total_bytes"] == 4000 and ag["eager"]["median_ms"] == pytest.approx(1.3)
    assert ag["eager"]["busbw_GBps"] == pytest.approx(B.algbw_GBps(4000, 1300) * 0.75)


def test_aggregate_failures_and_partial():
    ranks = [_rank(r, 1.0, ok=(r != 2)) for r in range(4)]
    ranks[1]["errors"]["allgather_graph"] = "RuntimeError('capture failed')"
    a = B.aggregate_ranks(ranks, 4)
    assert a["correct"] is False
    assert a["allgather_11_9MB"]["graph"]["status"] == "capture_failed"
    assert "capture failed" in a["allgather_11_9MB"]["graph"]["error"]["1"]
    p = B.aggregate_ranks([_rank(0, 1.0)], 4)
    assert p["correct"] is False and p["ranks_reported"] == [0]
    n = B.aggregate_ranks([], 4)
    assert n["correct"] is None and n["eager"] == {}


def _g(us, ok=True):
    return {"per_call_us": us, "graph_correct": ok}


def test_summary_rule():
    res = {"arms": {"baseline": {"status": "ok", "correct": True, "graph_983040": _g(300.0)},
                    "x": {"status": "ok", "correct": True, "graph_983040": _g(250.0)},
                    "y": {"status": "ok", "correct": False, "graph_983040": _g(100.0)}}}
    s = B.build_summary(res)
    assert s["best_arm"] == "x"                                  # y is fastest but its sum is wrong
    assert s["improvement_vs_baseline_pct"] == pytest.approx(100 / 6)
    assert s["adopt_candidate"] is None                          # 16.7 % < 20 %
    res["arms"]["x"]["graph_983040"]["per_call_us"] = 240.0
    assert B.build_summary(res)["adopt_candidate"] == "x"


@pytest.mark.parametrize("bad", [
    {"status": "ok", "correct": True, "graph_983040": _g(30.0, ok=False)},     # eager right, captured graph wrong
    {"status": "ok", "correct": True, "graph_983040": _g(30.0, ok=None)},      # graph correctness never measured
    {"status": "timeout", "correct": True, "graph_983040": _g(30.0)},          # the arm timed out
    {"status": "ok", "correct": False, "graph_983040": _g(30.0)},              # a rank missing -> aggregate not correct
    {"status": "ok", "correct": None, "graph_983040": _g(30.0)},               # no rank reported at all
])
def test_summary_rejects_unsafe_fast_arm(bad):
    """GPT audit 2 P0: a 90 % faster arm is never a candidate unless status ok + eager correct + graph correct."""
    res = {"arms": {"baseline": {"status": "ok", "correct": True, "graph_983040": _g(300.0)}, "fast": bad}}
    s = B.build_summary(res)
    assert s["best_arm"] == "baseline" and s["adopt_candidate"] is None, s


# ============================================================================================ bench: real child code, stub torch
def make_stub_torch(world=4, peer=True):
    """Minimal stand-in for the torch API the real child path calls (NOT torch semantics: no GPU, no NCCL).
    It proves the real-path Python runs end to end on CPU (names, call shapes, phase bookkeeping)."""
    import contextlib
    import types
    state = {"capture": None, "rank": 0}

    class Item:
        def __init__(self, v):
            self.v = v

        def item(self):
            return self.v

    class FT:
        def __init__(self, n, v=0.0):
            self.n, self.v = n, float(v)

        def min(self):
            return Item(self.v)

        def max(self):
            return Item(self.v)

        def fill_(self, v):
            self.v = float(v)
            return self

    def _n(shape):
        return shape if isinstance(shape, int) else shape[0]

    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-stub"
    torch.bfloat16, torch.float32 = "bf16", "f32"
    torch.device = lambda kind, idx=None: (kind, idx)
    torch.zeros = lambda shape, dtype=None, device=None: FT(_n(shape), 0)
    torch.ones = lambda shape, dtype=None, device=None: FT(_n(shape), 1)
    torch.empty = lambda shape, dtype=None, device=None: FT(_n(shape), 0)
    torch.full = lambda shape, val, dtype=None, device=None: FT(_n(shape), val)
    torch.version = types.SimpleNamespace(cuda="13.0-stub")
    torch._C = types.SimpleNamespace(_cuda_getDriverVersion=lambda: 13000)

    cuda = types.ModuleType("torch.cuda")

    class Event:
        def __init__(self, enable_timing=False):
            self.t = None

        def record(self):
            self.t = time.perf_counter()

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return (other.t - self.t) * 1000.0

    class Stream:
        def wait_stream(self, s):
            pass

    class CUDAGraph:
        def __init__(self):
            self.ops = []

        def replay(self):
            for f in self.ops:
                f()

    @contextlib.contextmanager
    def graph(g, capture_error_mode="global"):
        state["capture"] = g.ops
        try:
            yield
        finally:
            state["capture"] = None

    cuda.Event, cuda.Stream, cuda.CUDAGraph, cuda.graph = Event, Stream, CUDAGraph, graph
    cuda.synchronize = lambda: None
    cuda.set_device = lambda r: state.__setitem__("rank", r)
    cuda.get_device_name = lambda r=0: "STUB L4"
    cuda.current_stream = lambda: Stream()
    cuda.stream = lambda s: contextlib.nullcontext()
    cuda.device_count = lambda: world
    cuda.get_device_properties = lambda i: types.SimpleNamespace(name="STUB L4", major=8, minor=9,
                                                                 total_memory=24 * 2 ** 30, multi_processor_count=58)
    cuda.can_device_access_peer = lambda i, j: peer
    cuda.nccl = types.SimpleNamespace(version=lambda: (2, 29, 7))
    torch.cuda = cuda

    dist = types.ModuleType("torch.distributed")
    dist._init = False

    def init_process_group(backend=None, init_method=None, rank=None, world_size=None, device_id=None):
        assert backend == "nccl" and init_method.startswith("tcp://127.0.0.1:")
        dist._init = True

    def _sum(t):
        if t.v == state["rank"] + 1:
            t.v = world * (world + 1) / 2.0

    def all_reduce(t):
        if state["capture"] is not None:
            state["capture"].append(lambda: _sum(t))
        else:
            _sum(t)

    def all_gather_into_tensor(out, inp):
        if state["capture"] is not None:
            state["capture"].append(lambda: None)

    dist.init_process_group, dist.all_reduce, dist.all_gather_into_tensor = (init_process_group, all_reduce,
                                                                             all_gather_into_tensor)
    dist.is_initialized = lambda: dist._init
    dist.destroy_process_group = lambda: setattr(dist, "_init", False)
    torch.distributed = dist

    mp = types.ModuleType("torch.multiprocessing")

    def spawn(fn, args=(), nprocs=1, join=True):
        for i in range(nprocs):
            fn(i, *args)

    mp.spawn = spawn
    torch.multiprocessing = mp
    return {"torch": torch, "torch.cuda": cuda, "torch.distributed": dist, "torch.multiprocessing": mp}


def _install_stub(monkeypatch, **kw):
    for k, v in make_stub_torch(**kw).items():
        monkeypatch.setitem(sys.modules, k, v)


def test_real_child_path_with_stub_torch(tmp_path, monkeypatch):
    _install_stub(monkeypatch)
    wd = tmp_path / "arm"
    wd.mkdir()
    cfg = {"warmup": 1, "iters": 3, "graph_calls": 5, "replays": 2, "ag_bytes": 4096, "capture_mode": "thread_local"}
    ns = argparse.Namespace(child_fake=False, workdir=str(wd), cfg_json=json.dumps(cfg), arm="baseline")
    assert B.child_run(ns) == 0
    launcher = json.loads((wd / "launcher.json").read_text(encoding="utf-8"))
    assert launcher["error"] is None and launcher["world"] == 4 and launcher["port"] > 0
    ranks = [json.loads((wd / ("rank_%d.json" % r)).read_text(encoding="utf-8")) for r in range(4)]
    for rk in ranks:
        assert rk["errors"] == {}, rk["errors"]
        assert rk["phases_done"] == ["init", "correct", "eager", "graph_983040", "allgather_eager", "allgather_graph"]
        assert rk["correct"]["pass"] is True and rk["correct"]["expected"] == 10.0
        assert set(rk["eager"]) == {l for l, _ in B.EAGER_SIZES}
        assert len(rk["eager"]["983040B"]["samples_us"]) == 3
        g = rk["graph_983040"]
        assert g["calls"] == 5 and len(g["replay_us"]) == 2 and g["graph_correct"] is True
        assert g["capture_mode"] == "thread_local"
        assert len(rk["allgather_graph"]["samples_us"]) == 3 and rk["allgather_eager"]["bytes_per_rank"] == 4096
    agg = B.aggregate_ranks(ranks, 4)
    assert agg["correct"] is True and agg["graph_983040"]["calls_per_graph"] == 5
    assert agg["allgather_11_9MB"]["graph"]["status"] == "ok"


def test_real_child_requires_four_gpus(tmp_path, monkeypatch):
    _install_stub(monkeypatch, world=2)
    wd = tmp_path / "arm"
    wd.mkdir()
    ns = argparse.Namespace(child_fake=False, workdir=str(wd), cfg_json="{}", arm="baseline")
    assert B.child_run(ns) == 1
    assert "expected 4 GPUs" in json.loads((wd / "launcher.json").read_text(encoding="utf-8"))["error"]
    ns2 = argparse.Namespace(child_fake=False, workdir=str(wd), cfg_json=json.dumps({"world": 2, "iters": 2,
                             "replays": 1, "graph_calls": 2, "warmup": 1, "ag_bytes": 64}), arm="baseline")
    assert B.child_run(ns2) == 0                                 # --world 2 overrides the 4-GPU requirement


def test_real_probe_with_stub_torch(tmp_path, monkeypatch):
    _install_stub(monkeypatch, peer=False)
    out = tmp_path / "probe.json"
    assert B.child_probe(argparse.Namespace(child_out=str(out))) == 0
    p = json.loads(out.read_text(encoding="utf-8"))
    assert p["nccl"] == "2.29.7" and p["device_count"] == 4 and p["driver_version"] == 13000
    assert p["devices"][0]["capability"] == "8.9"
    assert p["can_access_peer"] == [[False] * 4 for _ in range(4)]


# ============================================================================================ trace summarizer
STEPS = 10
AR = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
AG = "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
MARLIN = "void marlin::Marlin<(vllm::ScalarTypeId)5, 256, 1, 8, 8, true, 2, 8, true>(int4 const*, int4 const*)"
ATTN = "void flashinfer::BatchDecodeWithPagedKVCacheKernel<(flashinfer::PosEncodingMode)0, 2u, 4u, 8u, 16u>()"
GDN1 = "fused_recurrent_gated_delta_rule_fwd_kernel"
GDN2 = "_causal_conv1d_update_kernel"
GEMM = "sm89_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x32_stage4_warpsize2x2x1_tensor16x8x16_kernel"
SAMP = "void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, (cub::BlockScanAlgorithm)2, 4u, true>()"
NORM = "void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16*)"
ELEM1 = "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >"
ELEM2 = "void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &silu>(__nv_bfloat16*)"
OTHER = "my_custom_unknown_kernel"

# (name, cat, calls per step, dur_us(rank))
KERNELS = [
    (AR, "kernel", 140, lambda r: 10.0 + 5.0 * r),
    (AG, "kernel", 1, lambda r: 2000.0 + 100.0 * r),
    (MARLIN, "kernel", 64, lambda r: 250.0),
    (ATTN, "kernel", 16, lambda r: 50.0),
    (GDN1, "kernel", 48, lambda r: 30.0),
    (GDN2, "kernel", 48, lambda r: 5.0),
    (GEMM, "kernel", 2, lambda r: 400.0),
    (SAMP, "kernel", 1, lambda r: 120.0),
    (NORM, "kernel", 129, lambda r: 4.0),
    (ELEM1, "kernel", 200, lambda r: 2.0),
    (ELEM2, "kernel", 64, lambda r: 6.0),
    ("Memcpy DtoD (Device -> Device)", "gpu_memcpy", 10, lambda r: 3.0),
    ("Memset (Device)", "gpu_memset", 2, lambda r: 1.0),
    (OTHER, "kernel", 1, lambda r: 7.0),
]
EXPECTED_MS = {  # per step, per rank r
    "nccl_allreduce": lambda r: 140 * (10 + 5 * r) / 1000, "nccl_allgather": lambda r: (2000 + 100 * r) / 1000,
    "nccl_other": lambda r: 0.0, "marlin_gemm": lambda r: 16.0, "attention": lambda r: 0.8,
    "gdn_triton": lambda r: 1.68, "other_gemm": lambda r: 0.8, "sampling": lambda r: 0.12, "norm": lambda r: 0.516,
    "memcpy": lambda r: 0.03, "memset": lambda r: 0.002, "elementwise_other": lambda r: 0.784,
    "other": lambda r: 0.007,
}


def make_trace(rank, steps=STEPS, verify=False):
    ev, ts = [], 1_000_000.0
    for s in range(steps):
        step_start = ts
        for name, cat, n, durf in KERNELS:
            d = durf(rank)
            for _ in range(n):
                ev.append({"ph": "X", "cat": cat, "name": name, "pid": rank, "tid": 7, "ts": ts, "dur": d,
                           "args": {"device": rank, "stream": 7, "correlation": len(ev)}})
                ts += d + 1.0
        ev.append({"ph": "X", "cat": "user_annotation", "name": "step[%s bs=96]" % ("VERIFY" if verify else "DECODE"),
                   "pid": 1, "tid": 1, "ts": step_start - 50, "dur": ts - step_start + 10})
        if verify:
            ev.append({"ph": "X", "cat": "user_annotation", "name": "step[DRAFT bs=96]", "pid": 1, "tid": 1,
                       "ts": step_start - 40, "dur": 5})
        ev.append({"ph": "X", "cat": "gpu_user_annotation", "name": "step[DECODE bs=96]", "pid": rank, "tid": 7,
                   "ts": step_start, "dur": ts - step_start})
        ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "pid": 1, "tid": 1, "ts": step_start,
                   "dur": 9999.0})
        ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::copy_", "pid": 1, "tid": 1, "ts": step_start, "dur": 5})
        ev.append({"ph": "s", "cat": "ac2g", "name": "ac2g", "id": s, "pid": 1, "tid": 1, "ts": step_start})
    return {"schemaVersion": 1, "deviceProperties": [{"id": rank, "name": "NVIDIA L4"}],
            "distributedInfo": {"backend": "nccl", "rank": rank, "world_size": 4},
            "record_shapes": 1, "traceEvents": ev, "traceName": "x", "displayTimeUnit": "ms",
            "baseTimeNanoseconds": 1729700000000000000}


def write_trace(path, obj):
    data = json.dumps(obj).encode("utf-8")
    if str(path).endswith(".gz"):
        with gzip.open(path, "wb") as f:
            f.write(data)
    else:
        Path(path).write_bytes(data)


@pytest.fixture
def trace_dir(tmp_path):
    d = tmp_path / "prof"
    d.mkdir()
    pid = "1729700000.5"
    for r in range(4):
        ext = ".trace.json.gz" if r < 2 else ".trace.json"           # gz and plain in one group
        write_trace(d / ("%s-TP-%d%s" % (pid, r, ext)), make_trace(r))
    (d / ("%s-TP-0-memory.pickle" % pid)).write_bytes(b"not a trace")
    write_trace(d / ("merged-%s.trace.json.gz" % pid), make_trace(0))
    return d


def test_classify_realistic_names():
    cases = {
        AR: "nccl_allreduce", "ncclKernel_AllReduce_RING_LL_Sum_half": "nccl_allreduce",
        "cross_device_reduce_1stage<__nv_bfloat16, 4>": "nccl_allreduce", AG: "nccl_allgather",
        "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)": "nccl_other", MARLIN: "marlin_gemm",
        "gptq_marlin_repack_kernel": "marlin_gemm", ATTN: "attention",
        "void flashinfer::BatchPrefillWithPagedKVCacheKernel<...>": "attention",
        "void flashinfer::MergeStatesKernel<8u, 16u>": "attention", "_fwd_kernel_stage1 attention": "attention",
        "fused_sigmoid_gating_delta_rule_update_kernel": "gdn_triton", GDN1: "gdn_triton", GDN2: "gdn_triton",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "gdn_triton", "chunk_local_cumsum_scalar_kernel": "gdn_triton",
        GEMM: "other_gemm", "ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn": "other_gemm",
        "void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_32x32_128x1_tn_align8>": "other_gemm",
        "void cublasLt::splitKreduce_kernel<32, 16, int, float>": "other_gemm",
        SAMP: "sampling", "void at::native::radixSortKVInPlace<-2, -1, 32, 4, float, long>": "sampling",
        "void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float>": "sampling",
        "argmax_kernel": "sampling", NORM: "norm", "rmsnorm_kernel": "norm", "l2norm_fwd_kernel": "norm",
        ELEM1: "elementwise_other", ELEM2: "elementwise_other",
        "void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheKernel<...>": "elementwise_other",
        "void at::native::index_elementwise_kernel<128, 4>": "elementwise_other",
        "void flashinfer::some_new_kernel<1>": "attention", OTHER: "other",
    }
    for name, cat in cases.items():
        assert T.classify(name) == cat, name
    assert T.classify("Memcpy HtoD (Pinned -> Device)", "gpu_memcpy") == "memcpy"
    assert T.classify("Memset (Device)", "gpu_memset") == "memset"


def test_summarize_exact_ms_per_step(trace_dir):
    s = T.summarize(str(trace_dir), STEPS, top=5)
    assert s["files_read"] == 4
    assert [x for x in s["skipped"] if "merged" in x["reason"]]
    assert list(s["groups"]) == ["1729700000.5-TP-*"] and s["main_group"] == "1729700000.5-TP-*"
    g = s["groups"]["1729700000.5-TP-*"]
    assert g["n_ranks"] == 4 and g["num_steps"] == STEPS and g["num_steps_source"] == "argument"
    assert g["detected_steps"] == {"0": STEPS, "1": STEPS, "2": STEPS, "3": STEPS}
    for r in range(4):
        rk = g["ranks"][str(r)]
        total = 0.0
        for c, f in EXPECTED_MS.items():
            assert rk["categories"][c]["ms_per_step"] == pytest.approx(f(r), abs=1e-9), (r, c)
            total += f(r)
        assert rk["kernel_sum_ms_per_step"] == pytest.approx(total)
        assert rk["gpu_busy_ms_per_step"] == pytest.approx(total)          # no overlap in this trace
        n_kern = sum(k[2] for k in KERNELS)
        assert rk["wall_ms_per_step"] == pytest.approx((total * 1000 * STEPS + n_kern * STEPS - 1) / 1000 / STEPS)
        assert rk["categories"]["nccl_allreduce"]["calls_per_step"] == 140
        assert sum(v["share_of_kernel_sum"] for v in rk["categories"].values()) == pytest.approx(1.0)
        assert rk["step_spans"]["DECODE"]["count"] == STEPS
    ar = g["collectives"]["nccl_allreduce"]
    assert ar["min_over_ranks_ms_per_step"] == pytest.approx(1.4) and ar["min_rank"] == "0"
    assert ar["max_over_ranks_ms_per_step"] == pytest.approx(3.5) and ar["max_rank"] == "3"
    assert ar["skew_ms_per_step"] == pytest.approx(2.1)
    agc = g["collectives"]["nccl_allgather"]
    assert agc["min_over_ranks_ms_per_step"] == pytest.approx(2.0) and agc["skew_ms_per_step"] == pytest.approx(0.3)
    assert g["categories"]["nccl_allreduce"]["ms_per_step_mean"] == pytest.approx((1.4 + 2.1 + 2.8 + 3.5) / 4)
    top = g["top_kernels"]
    assert len(top) == 5 and top[0]["name"].startswith("void marlin::Marlin")
    assert top[0]["category"] == "marlin_gemm" and top[0]["count_total"] == 64 * STEPS * 4
    assert top[0]["ms_per_step_per_rank"] == pytest.approx(16.0) and top[0]["mean_us"] == pytest.approx(250.0)
    assert top[0]["calls_per_step_per_rank"] == pytest.approx(64)
    assert top[1]["category"] == "nccl_allreduce" and top[1]["ms_per_step_per_rank"] == pytest.approx(2.45)
    json.dumps(s)                                                         # JSON-serializable
    assert "marlin_gemm" in T.format_table(s)


def test_detected_steps_and_verify(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    for r in range(2):
        write_trace(d / ("pref-777.1-TP-%d-DECODE.trace.json.gz" % r), make_trace(r, steps=4, verify=True))
        write_trace(d / ("pref-777.1-TP-%d-EXTEND.trace.json.gz" % r), make_trace(r, steps=2))
    s = T.summarize(str(d), 0)
    assert set(s["groups"]) == {"pref-777.1-TP-*-DECODE", "pref-777.1-TP-*-EXTEND"}
    assert s["main_group"] == "pref-777.1-TP-*-DECODE"
    gd = s["groups"]["pref-777.1-TP-*-DECODE"]
    assert gd["num_steps"] == 4 and gd["num_steps_source"] == "detected step[] spans"
    assert gd["ranks"]["0"]["step_spans"]["DRAFT"]["count"] == 4
    assert gd["ranks"]["1"]["categories"]["marlin_gemm"]["ms_per_step"] == pytest.approx(16.0)
    ge = s["groups"]["pref-777.1-TP-*-EXTEND"]
    assert ge["num_steps"] == 2


def test_union_busy_with_overlap(tmp_path):
    ev = [{"ph": "X", "cat": "kernel", "name": "a_gemm", "ts": 0.0, "dur": 10.0},
          {"ph": "X", "cat": "kernel", "name": AR, "ts": 5.0, "dur": 10.0},
          {"ph": "X", "cat": "kernel", "name": "b_gemm", "ts": 20.0, "dur": 10.0},
          {"ph": "X", "cat": "kernel", "name": "inner", "ts": 21.0, "dur": 2.0}]
    p = tmp_path / "x-TP-0.trace.json"
    write_trace(p, {"traceEvents": ev})
    g = T.summarize(str(p), 1)["groups"]["x-TP-*"]
    assert g["ranks"]["0"]["kernel_sum_ms_per_step"] == pytest.approx(0.032)
    assert g["ranks"]["0"]["gpu_busy_ms_per_step"] == pytest.approx(0.025)
    assert g["ranks"]["0"]["wall_ms_per_step"] == pytest.approx(0.030)
    assert g["ranks"]["0"]["categories"]["nccl_allreduce"]["share_of_busy"] == pytest.approx(0.4)


def test_size_guard_and_bad_files(trace_dir, tmp_path):
    s = T.summarize(str(trace_dir), STEPS, max_file_mb=1e-6)
    assert s["files_read"] == 0 and s["groups"] == {} and s["main_group"] is None
    guarded = [x for x in s["skipped"] if "max_file_mb" in x["reason"]]
    assert len(guarded) == 4
    d = tmp_path / "bad"
    d.mkdir()
    full = gzip.compress(json.dumps(make_trace(0, steps=2)).encode())
    (d / "9.9-TP-0.trace.json.gz").write_bytes(full[: len(full) // 2])       # truncated gzip
    (d / "9.9-TP-1.trace.json").write_text('{"traceEvents": [{"ph": "X", "cat": "kernel", "name": "k", "ts": 0, "du',
                                           encoding="utf-8")                    # truncated JSON
    write_trace(d / "9.9-TP-2.trace.json", make_trace(2, steps=2))
    s2 = T.summarize(str(d), 2)
    assert s2["files_read"] == 1
    assert len([x for x in s2["skipped"] if "read failed" in x["reason"]]) == 2
    assert list(s2["groups"]["9.9-TP-*"]["ranks"]) == ["2"]
    assert T.summarize(str(tmp_path / "nope"), 1)["skipped"][0]["reason"] == "not a file or directory"


def test_stream_parser_matches_json_load(tmp_path):
    obj = make_trace(1, steps=2)
    obj["traceEvents"].append({"ph": "X", "cat": "kernel", "name": 'we"ird \\ name é [ ] { }', "ts": 1, "dur": 2})
    p = tmp_path / "s.json"
    write_trace(p, obj)
    for chunk in (7, 64, 1 << 20):
        got = list(T.iter_trace_events(str(p), chunk_chars=chunk))
        assert got == obj["traceEvents"], chunk
    pa = tmp_path / "arr.json"
    pa.write_text(json.dumps(obj["traceEvents"][:5]) + "\n", encoding="utf-8")
    assert list(T.iter_trace_events(str(pa), chunk_chars=16)) == obj["traceEvents"][:5]
    pn = tmp_path / "none.json"
    pn.write_text('{"foo": 1}', encoding="utf-8")
    assert list(T.iter_trace_events(str(pn))) == []
    pt = tmp_path / "trail.json"                                    # tolerate a trailing comma before ]
    pt.write_text('{"traceEvents": [{"ph": "X"}, {"ph": "i"},\n]}', encoding="utf-8")
    assert len(list(T.iter_trace_events(str(pt), chunk_chars=5))) == 2


def test_rank_from_distributed_info_when_no_tp_in_name(tmp_path):
    d = tmp_path / "anon"
    d.mkdir()
    write_trace(d / "worker_a.json", make_trace(3, steps=1))
    write_trace(d / "worker_b.json.gz", make_trace(1, steps=1))
    g = T.summarize(str(d), 1)["groups"]["ungrouped"]
    assert set(g["ranks"]) == {"1", "3"}
    assert g["ranks"]["3"]["categories"]["nccl_allreduce"]["ms_per_step"] == pytest.approx(140 * 25 / 1000)


def test_cli_writes_summary(trace_dir, tmp_path):
    out = tmp_path / "summary.json"
    p = subprocess.run([PY, SUMMARY, str(trace_dir), "--steps", str(STEPS), "--out", str(out), "--top", "3"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    s = json.loads(out.read_text(encoding="utf-8"))
    g = s["groups"][s["main_group"]]
    assert len(g["top_kernels"]) == 3
    assert g["collectives"]["nccl_allreduce"]["min_over_ranks_ms_per_step"] == pytest.approx(1.4)
    assert "nccl_allreduce" in p.stdout


def test_summary_invalid_baseline_is_no_reference():
    """Review 10: a baseline that timed out or is graph-wrong cannot be the reference for improvement / adoption."""
    res = {"arms": {"baseline": {"status": "timeout", "correct": True, "graph_983040": _g(300.0)},
                    "x": {"status": "ok", "correct": True, "graph_983040": _g(100.0)}}}
    s = B.build_summary(res)
    assert s["baseline_per_call_us"] is None and s["adopt_candidate"] is None
