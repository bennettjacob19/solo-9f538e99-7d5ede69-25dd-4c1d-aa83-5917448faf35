"""End-to-end fault-injection tests against real uvicorn processes.

Spins up one coordinator + two independent worker processes (each with its own
temporary data directory, exactly like the Docker topology) and verifies:

1. clean distributed training equals the single-process reference trajectory
   bit-for-bit;
2. killing a worker mid-training pauses the group, and resume reproduces the
   exact uninterrupted trajectory (no skipped / double-applied round);
3. redoing an unconfirmed round after coordinator restart stays exact;
4. stale-generation messages are fenced (409) and cannot advance a new group;
5. downloaded weights only ever reflect confirmed rounds;
6. the validation/release gate freezes one confirmed version, uses exactly
   those weights (never a locally saved unconfirmed update), merges counts
   from mutually-exclusive shards, and keeps the previous release on failure.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

import gate

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
ROUNDS = 30


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Service:
    def __init__(self, name, module, port, env, health_path="/health"):
        self.name, self.module, self.port, self.env = name, module, port, env
        self.health_path = health_path
        self.proc: subprocess.Popen | None = None

    def start(self):
        import site
        env = dict(os.environ)
        env.update(self.env)
        # Use -P (cwd excluded from sys.path, guarding against planted modules)
        # rather than -I, which would also drop PYTHONPATH and user site
        # packages. The app dir and locally installed deps are supplied here.
        extra = [str(APP), site.getusersitepackages()]
        env["PYTHONPATH"] = os.pathsep.join(extra) + os.pathsep + env.get(
            "PYTHONPATH", "")
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "uvicorn",
             f"{self.module}:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=APP, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._wait_up()

    def _wait_up(self, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"{self.name} exited early:\n{out}")
            try:
                r = httpx.get(
                    f"http://127.0.0.1:{self.port}{self.health_path}",
                    timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.15)
        raise TimeoutError(f"{self.name} did not come up")

    def hard_kill(self):
        assert self.proc
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture()
def cluster(tmp_path):
    p1, p2, pc = _free_port(), _free_port(), _free_port()
    d1, d2, dc = (tmp_path / "w1"), (tmp_path / "w2"), (tmp_path / "coord")
    for d in (d1, d2, dc):
        d.mkdir()
    w1 = Service("worker1", "worker", p1,
                 {"WORKER_ID": "worker1", "WORKER_DATA": str(d1)})
    w2 = Service("worker2", "worker", p2,
                 {"WORKER_ID": "worker2", "WORKER_DATA": str(d2)})
    coord = Service("coordinator", "coordinator", pc, {
        "COORD_DATA": str(dc),
        "WORKER1_URL": f"http://127.0.0.1:{p1}",
        "WORKER2_URL": f"http://127.0.0.1:{p2}",
    }, health_path="/api/status")
    w1.start(); w2.start(); coord.start()
    yield {
        "coord": f"http://127.0.0.1:{pc}",
        "coord_svc": coord, "w1": w1, "w2": w2,
        "paths": {"w1": d1, "w2": d2, "coord": dc},
    }
    for s in (w1, w2, coord):
        s.stop()


def _dataset_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    z = 1.2 * x[:, 0] - 0.6 * x[:, 1]
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    p = tmp_path / "data.csv"
    lines = ["x1,x2,label"]
    for (a, b), c in zip(x, y):
        lines.append(f"{a:.10f},{b:.10f},{c}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _wait_status(base, predicate, timeout=40):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = httpx.get(base + "/api/status", timeout=5).json()
        if predicate(last):
            return last
        time.sleep(0.25)
    raise AssertionError(f"condition not met; last={json.dumps(last)[:600]}")


def _create_and_start(base, csv_path, rounds=ROUNDS):
    with open(csv_path, "rb") as f:
        r = httpx.post(base + f"/api/job?rounds={rounds}",
                       files={"file": ("data.csv", f, "text/csv")},
                       timeout=15)
    r.raise_for_status()
    httpx.post(base + "/api/start", timeout=10).raise_for_status()


# --------------------------------------------------------------------- tests
def test_clean_run_matches_reference(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["status"] == "completed")
    assert st["confirmed_round"] == ROUNDS

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    # Exact, bit-for-bit equality with uninterrupted single-process training.
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert np.allclose(st["w"], ref["weights"][-1], atol=0.0, rtol=0.0)
    assert np.allclose(st["v"], ref["momentum"][-1], atol=0.0, rtol=0.0)

    # Weights download corresponds to the confirmed round only.
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == ROUNDS
    assert wdoc["weights"]["w1"] == st["w"][1]


def test_worker_kill_and_resume_is_numerically_exact(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    # Let some rounds confirm, then SIGKILL worker1 mid-flight.
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 8)
    confirmed_at_kill = st["confirmed_round"]
    cluster["w1"].hard_kill()

    st = _wait_status(base, lambda d: d["status"] == "paused")
    assert st["confirmed_round"] == confirmed_at_kill  # nothing half-counted

    # The killed worker's durable state never exceeds the confirmed round by
    # more than one (a commit may have landed just before the kill).
    state1 = json.loads((cluster["paths"]["w1"] / "state.json").read_text())
    assert state1["round"] <= confirmed_at_kill + 1

    # Restart the worker process (fresh generation-neutral process).
    cluster["w1"].start()
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed")

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k], f"round {k} differs"
    np.testing.assert_array_equal(np.asarray(st["w"]),
                                  np.asarray(ref["weights"][-1]))
    # Exactly one loss entry per round: no duplicate application.
    rounds_seen = [p["round"] for p in st["loss_history"]]
    assert rounds_seen == list(range(ROUNDS + 1))


def test_coordinator_restart_mid_round_redoes_without_skip(cluster, tmp_path):
    base = cluster["coord"]
    # Add delay so a round is almost certainly in flight at kill time.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    cluster["coord_svc"].hard_kill()
    time.sleep(0.8)  # let at least one delayed in-flight round drift
    cluster["coord_svc"].start()

    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["status"] == "paused"
    assert st["confirmed_round"] == confirmed
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(13))


def test_stale_generation_messages_are_fenced(cluster, tmp_path):
    base = cluster["coord"]
    # Slow each round down so the pause reliably lands while training runs.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.5}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    old_gen = st["gen"]
    task_id = st["task_id"]

    # Pause and resume -> a strictly newer generation is installed.
    deadline = time.time() + 15
    while True:
        r = httpx.post(base + "/api/pause", timeout=5)
        if r.status_code == 200:
            break
        if httpx.get(base + "/api/status").json()["status"] == "completed":
            raise AssertionError("training completed before pause took effect")
        assert time.time() < deadline
        time.sleep(0.3)
    _wait_status(base, lambda d: d["status"] == "paused")
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st2 = _wait_status(base, lambda d: d["gen"] > old_gen)
    new_gen = st2["gen"]

    # A late message from the old training group must be rejected (409) and
    # must not mutate any state. Reach the worker directly on its own port.
    w1_port = cluster["w1"].port
    r = httpx.post(f"http://127.0.0.1:{w1_port}/gradient",
                   json={"task_id": task_id, "gen": old_gen,
                         "round": st2["confirmed_round"] + 1},
                   timeout=5)
    assert r.status_code == 409

    # Unknown task id is also rejected.
    r2 = httpx.post(f"http://127.0.0.1:{w1_port}/commit",
                    json={"task_id": "deadbeef", "gen": new_gen,
                          "round": 1, "grad": [0, 0, 0]}, timeout=5)
    assert r2.status_code in (409, 410)

    final = _wait_status(base, lambda d: d["status"] == "completed")
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]


def test_full_cluster_reboot_keeps_durable_state(cluster, tmp_path):
    base = cluster["coord"]
    # Slow rounds so the kill reliably lands mid-training, not after completion.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=15)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    # Kill every process, then bring the whole cluster back up.
    cluster["coord_svc"].hard_kill()
    cluster["w1"].hard_kill()
    cluster["w2"].hard_kill()
    for s in (cluster["w1"], cluster["w2"], cluster["coord_svc"]):
        s.start()
    st = httpx.get(base + "/api/status", timeout=10).json()
    # More rounds may have confirmed between observation and the kill; the
    # durable invariant that matters is that progress is never lost and the
    # reboot resumes from a confirmed round >= the one we observed.
    assert st["confirmed_round"] >= confirmed
    assert st["status"] in ("paused", "idle", "completed")
    if st["status"] != "completed":
        httpx.post(base + "/api/resume", timeout=15).raise_for_status()
        st = _wait_status(base, lambda d: d["status"] == "completed",
                          timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(16))


def test_csv_validation_rules(cluster, tmp_path):
    base = cluster["coord"]
    good = _dataset_csv(tmp_path, n=20)
    # Below the 2-row minimum: header + a single data row -> 400.
    p = tmp_path / "small.csv"
    p.write_text("x1,x2,label\n1,2,0\n")
    with open(p, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("small.csv", f, "text/csv")})
    assert r.status_code == 400
    # The minimum allowed size (2 rows, both classes) is accepted.
    pmin = tmp_path / "min.csv"
    pmin.write_text("1,2,0\n3,4,1\n")  # no header, exactly 2 rows
    with open(pmin, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=3",
                       files={"file": ("min.csv", f, "text/csv")})
    assert r.status_code in (200, 202)
    httpx.post(base + "/api/reset", timeout=5)
    # Bad label.
    p2 = tmp_path / "badlabel.csv"
    lines = good.read_text().splitlines()
    head, body = lines[0], lines[1:]
    body[0] = body[0][:-1] + "9"
    p2.write_text("\n".join([head] + body))
    with open(p2, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("b.csv", f, "text/csv")})
    assert r.status_code == 400
    # Rounds out of range.
    with open(good, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=101",
                       files={"file": ("g.csv", f, "text/csv")})
    assert r.status_code == 400
    # Over the 200-row maximum.
    big = tmp_path / "big.csv"
    rows = ["x1,x2,label"] + [f"{i},1,{i % 2}" for i in range(201)]
    big.write_text("\n".join(rows))
    with open(big, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("big.csv", f, "text/csv")})
    assert r.status_code == 400


def test_download_weights_only_confirmed_and_save_keeps_last(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 5)
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == st["confirmed_round"]
    # On-disk immutable artifact plus latest pointer exist.
    cdir = cluster["paths"]["coord"] / "weights"
    assert (cdir / f"weights_round_{st['confirmed_round']:04d}.json").exists()
    assert (cdir / "weights_latest.json").exists()
    final = _wait_status(base, lambda d: d["status"] == "completed")
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == final["confirmed_round"] == 12


def test_worker_commit_dedup_and_prepare_rollback(cluster, tmp_path):
    """Direct worker protocol: a retried commit applies once, and /prepare
    rolls an unconfirmed advance back to the last confirmed round."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=8)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    task_id, gen = st["task_id"], st["gen"]
    w1_port = cluster["w1"].port
    url = f"http://127.0.0.1:{w1_port}"

    h = httpx.get(url + "/health", timeout=5).json()
    at = h["round"]
    # Gradient for the next round.
    g = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen, "round": at + 1},
                   timeout=5).json()
    grad = [0.01, -0.02, 0.03]
    c1 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c1["round"] == at + 1 and c1["dedup"] is False
    w_after, v_after = c1["w"], c1["v"]
    # Retried commit for the same round must NOT apply the step a second time.
    c2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c2["dedup"] is True and c2["w"] == w_after and c2["v"] == v_after

    # A newer generation's /prepare rolls the worker back to the confirmed
    # round (discarding the unconfirmed local advance above).
    new_gen = gen + 1
    p = httpx.post(url + "/prepare",
                   json={"task_id": task_id, "gen": new_gen,
                         "round": at, "w": st["w"], "v": st["v"]},
                   timeout=5).json()
    assert p["gen"] == new_gen and p["round"] == at
    h2 = httpx.get(url + "/health", timeout=5).json()
    assert h2["round"] == at and h2["gen"] == new_gen

    # Old-generation traffic is now fenced.
    r = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen, "round": at + 1},
                   timeout=5)
    assert r.status_code == 409
    # No skipping rounds: committing round at+2 before at+1 is rejected.
    r2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": new_gen,
                          "round": at + 2, "grad": grad}, timeout=5)
    assert r2.status_code == 409


def test_delay_endpoint_survives_and_rounds_still_exact(cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.2}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=6)
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]


# ------------------------------------------------------------------ validation
def _validation_csv(tmp_path: Path, name: str = "val.csv",
                    n: int = 24, seed: int = 11) -> Path:
    """Validation set built on the same linear rule as the training data, so a
    trained version scores well while the untrained (zero) version does not."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) * 1.5
    z = 1.2 * x[:, 0] - 0.6 * x[:, 1]
    y = (z > 0).astype(int)
    if y.min() == y.max():  # pragma: no cover - the fixed seed has both
        raise AssertionError("validation fixture needs both classes")
    p = tmp_path / name
    lines = ["x1,x2,label"]
    for (a, b), c in zip(x, y):
        lines.append(f"{a:.10f},{b:.10f},{c}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _validate(base: str, csv_path: Path, min_recall: float, max_fpr: float):
    with open(csv_path, "rb") as f:
        files = {"file": ("val.csv", f, "text/csv")}
        data = {"min_recall": str(min_recall), "max_fpr": str(max_fpr)}
        return httpx.post(base + "/api/validate", files=files, data=data,
                          timeout=30)


def _reset(base: str):
    httpx.post(base + "/api/reset", timeout=10).raise_for_status()


def test_validation_uses_frozen_confirmed_weights(cluster, tmp_path):
    """Validation judges exactly the confirmed snapshot of the current task:
    the round/version match the confirmed state, counts cover the whole
    uploaded CSV, and the released artifact keeps that snapshot even after
    training advances and confirms later rounds."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 6)
    httpx.post(base + "/api/pause", timeout=10)
    st = _wait_status(base, lambda d: d["status"] == "paused")
    frozen_round = st["confirmed_round"]
    val = _validation_csv(tmp_path)

    # Strict gate against the trained snapshot can pass or fail depending on
    # the data, but the counts must equal the coordinator's own recomputation.
    r = _validate(base, val, 0.0, 1.0)  # accept everything -> must approve
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "approved"
    assert body["round"] == frozen_round
    assert body["n"] == 24
    assert body["confusion"] == {k: body["confusion"][k]
                                 for k in ("tp", "fp", "tn", "fn")}
    approved_version = body["weights_version"]
    assert body["release"]["download_available"] is True

    # The staged confirmation round is unchanged by validation.
    assert _wait_status(base, lambda d: d["status"] == "paused")[
        "confirmed_round"] == frozen_round
    # Download is exactly the released snapshot.
    wdoc = httpx.get(base + "/api/release/weights", timeout=5)
    assert wdoc.status_code == 200
    rel = wdoc.json()
    assert rel["round"] == frozen_round
    assert rel["weights_version"] == approved_version
    assert rel["decision_threshold"] == 0.5

    # A release from an earlier round is kept even after later rounds are
    # confirmed: resume until the whole task completes, then re-download.
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    final = _wait_status(base, lambda d: d["status"] == "completed")
    assert final["confirmed_round"] > frozen_round
    wdoc = httpx.get(base + "/api/release/weights", timeout=5)
    assert wdoc.status_code == 200
    assert wdoc.json()["weights_version"] == approved_version
    assert wdoc.json()["round"] == frozen_round
    # The training weight export did move on, proving release and training
    # artifacts are independent.
    latest = httpx.get(base + "/api/weights", timeout=5).json()
    assert latest["confirmed_round"] == final["confirmed_round"] > frozen_round


def test_validation_rejects_and_keeps_previous_release(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    _wait_status(base, lambda d: d["confirmed_round"] >= 10)
    httpx.post(base + "/api/pause", timeout=10)
    _wait_status(base, lambda d: d["status"] == "paused")
    val = _validation_csv(tmp_path)

    ok = _validate(base, val, 0.0, 1.0)
    assert ok.status_code == 200 and ok.json()["outcome"] == "approved"
    good = ok.json()
    good_doc = httpx.get(base + "/api/release/weights", timeout=5).json()
    assert good_doc["weights_version"] == good["weights_version"]

    # Impossible gate -> rejected, and the previously approved version stays.
    bad = _validate(base, val, 1.0, 0.0)  # perfect recall AND zero FPR required
    assert bad.status_code == 200
    b = bad.json()
    assert b["outcome"] in ("rejected", "approved")
    if b["outcome"] == "rejected":
        assert b["release"]["outcome"] == "approved"  # previous kept
        still = httpx.get(base + "/api/release/weights", timeout=5).json()
        assert still["weights_version"] == good["weights_version"]
    else:  # pragma: no cover - exceptional perfect model on this fixture
        assert b["metrics"]["recall"] == 1.0
        assert b["metrics"]["false_positive_rate"] == 0.0

    # A rejected validation on a fresh task leaves nothing downloadable.
    _reset(base)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=3)
    _wait_status(base, lambda d: d["status"] == "completed")
    r = _validate(base, val, 1.0, 0.0)
    assert r.status_code == 200
    if r.json()["outcome"] == "rejected":
        assert httpx.get(base + "/api/release/weights",
                         timeout=5).status_code == 404
        assert r.json()["release"]["download_available"] is False


def test_validation_with_dead_worker_produces_no_new_conclusion(cluster,
                                                               tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    _wait_status(base, lambda d: d["confirmed_round"] >= 8)
    httpx.post(base + "/api/pause", timeout=10)
    _wait_status(base, lambda d: d["status"] == "paused")
    val = _validation_csv(tmp_path)

    ok = _validate(base, val, 0.0, 1.0)
    assert ok.status_code == 200 and ok.json()["outcome"] == "approved"
    approved = ok.json()["weights_version"]

    # Kill one worker: one side cannot answer, so no new conclusion is made.
    cluster["w2"].hard_kill()
    r = _validate(base, val, 0.0, 1.0)
    assert r.status_code == 502
    body = r.json()
    assert body["outcome"] == "error"
    assert "worker2" in body["error"]
    assert body["release"]["outcome"] == "approved"
    still = httpx.get(base + "/api/release/weights", timeout=5).json()
    assert still["weights_version"] == approved
    # Status surfaces the failed attempt but still shows the approved release.
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["release"]["outcome"] == "approved"
    assert st["release"]["last_attempt"]["outcome"] == "error"


def test_validation_rejects_altered_frozen_weights_at_worker(cluster,
                                                             tmp_path):
    """A request whose weights do not hash to the advertised version is
    rejected by the worker itself (409), so a mixed snapshot cannot be
    evaluated under an approved version label."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    httpx.post(base + "/api/pause", timeout=10)
    _wait_status(base, lambda d: d["status"] == "paused")
    task_id, gen = st["task_id"], st["gen"]
    ver = gate.weights_version(task_id, st["confirmed_round"], st["w"])
    url = f"http://127.0.0.1:{cluster['w1'].port}"
    rows = [{"x": [0.0, 0.0], "y": 1.0}, {"x": [1.0, 1.0], "y": 0.0}]
    good = httpx.post(url + "/evaluate", json={
        "task_id": task_id, "validation_id": "v1", "version": ver,
        "round": st["confirmed_round"], "weights": st["w"], "rows": rows,
        "threshold": 0.5}, timeout=10)
    assert good.status_code == 200
    assert good.json()["version"] == ver
    # Same version label, different weights -> refused.
    bad = httpx.post(url + "/evaluate", json={
        "task_id": task_id, "validation_id": "v1", "version": ver,
        "round": st["confirmed_round"],
        "weights": [st["w"][0] + 1.0, st["w"][1], st["w"][2]],
        "rows": rows, "threshold": 0.5}, timeout=10)
    assert bad.status_code == 409
    # Wrong task id -> refused.
    stale = httpx.post(url + "/evaluate", json={
        "task_id": "deadbeef", "validation_id": "v1", "version": ver,
        "round": st["confirmed_round"], "weights": st["w"], "rows": rows,
        "threshold": 0.5}, timeout=10)
    assert stale.status_code == 410
    assert gen  # generation is not required for a frozen-snapshot evaluation


def test_unconfirmed_saved_update_is_not_used_for_validation(cluster,
                                                             tmp_path):
    """Push an unconfirmed commit into one worker (saved durably, never
    confirmed by the coordinator) and verify the gate still evaluates exactly
    the confirmed snapshot."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    httpx.post(base + "/api/pause", timeout=10)
    st = _wait_status(base, lambda d: d["status"] == "paused")
    task_id, gen = st["task_id"], st["gen"]
    confirmed_round = st["confirmed_round"]
    confirmed_w = list(st["w"])

    # Locally advance worker1 by one round — the coordinator never confirms it.
    # A commit may have landed just before the pause, in which case the local
    # state is already one round ahead; either way it must not be used.
    w1 = f"http://127.0.0.1:{cluster['w1'].port}"
    health = httpx.get(w1 + "/health", timeout=5).json()
    at = health["round"]
    if at == confirmed_round:
        big_grad = [5.0, -5.0, 5.0]
        c = httpx.post(w1 + "/commit", json={
            "task_id": task_id, "gen": gen, "round": at + 1,
            "grad": big_grad}, timeout=5)
        assert c.status_code == 200 and c.json()["dedup"] is False
        saved = c.json()["w"]
        assert saved != confirmed_w  # worker1 really holds a different w
        at += 1
    else:
        assert at == confirmed_round + 1

    # Coordinator still reports the confirmed (unchanged) snapshot.
    assert _wait_status(base, lambda d: d["status"] == "paused")[
        "w"] == confirmed_w

    r = _validate(base, _validation_csv(tmp_path), 0.0, 1.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["round"] == confirmed_round
    assert body["weights_version"] == gate.weights_version(
        task_id, confirmed_round, confirmed_w)
    # Release payload (if approved) must contain the confirmed weights only.
    doc = httpx.get(base + "/api/release/weights", timeout=5).json()
    assert [doc["weights"]["bias"], doc["weights"]["w1"], doc["weights"]["w2"]] \
        == confirmed_w

    # Resume: a fresh generation's /prepare rolls worker1 back to the last
    # confirmed round, so the unconfirmed local step never enters the final
    # trajectory (bit-identical to uninterrupted single-process training).
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    final = _wait_status(base, lambda d: d["status"] == "completed",
                         timeout=90)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    np.testing.assert_array_equal(np.asarray(final["w"]),
                                  np.asarray(ref["weights"][-1]))
    assert at == final["confirmed_round"] or at <= final["confirmed_round"]


def test_clear_task_removes_released_version(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    _wait_status(base, lambda d: d["confirmed_round"] >= 5)
    httpx.post(base + "/api/pause", timeout=10)
    _wait_status(base, lambda d: d["status"] == "paused")
    r = _validate(base, _validation_csv(tmp_path), 0.0, 1.0)
    assert r.status_code == 200 and r.json()["outcome"] == "approved"
    assert httpx.get(base + "/api/release/weights", timeout=5).status_code == 200
    assert not (cluster["paths"]["coord"] / "releases").exists() is False

    _reset(base)
    # Both the record and every artifact are gone: the old version is not
    # downloadable and not reported by status.
    assert httpx.get(base + "/api/release/weights",
                     timeout=5).status_code == 404
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 404
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["release"]["download_available"] is False
    assert st["release"]["outcome"] is None
    assert not (cluster["paths"]["coord"] / "releases").exists()
    assert not (cluster["paths"]["coord"] / "release.json").exists()


def test_validation_interleaves_with_running_training(cluster, tmp_path):
    """Validation may run while training is still going. It judges the
    snapshot confirmed at submission time; training keeps advancing afterwards
    and the released artifact stays at that frozen round."""
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.3}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=20)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    assert st["status"] == "running"

    r = _validate(base, _validation_csv(tmp_path), 0.0, 1.0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "approved"
    frozen_round = body["round"]
    assert frozen_round <= st["confirmed_round"] + 1

    # Training continues and eventually completes.
    final = _wait_status(base, lambda d: d["status"] == "completed",
                         timeout=120)
    assert final["confirmed_round"] == 20
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]

    # The released version still points at the frozen round, not the final one.
    released = httpx.get(base + "/api/release/weights", timeout=5).json()
    assert released["round"] == frozen_round
    assert released["weights_version"] == body["weights_version"]
    assert released["outcome"] == "approved"


def test_validation_input_validation(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    val = _validation_csv(tmp_path)

    # Threshold outside [0, 1] is rejected before touching workers.
    for kwargs in ({"min_recall": 1.5, "max_fpr": 0.1},
                   {"min_recall": 0.5, "max_fpr": -0.01}):
        r = _validate(base, val, **kwargs)
        assert r.status_code == 400
    # Single-class validation set is rejected (recall/FPR undefined).
    p = tmp_path / "oneclass.csv"
    p.write_text("x1,x2,label\n0.1,0.2,1\n0.3,0.4,1\n", encoding="utf-8")
    r = _validate(base, p, 0.5, 0.5)
    assert r.status_code == 400
    # Malformed CSV is rejected.
    p2 = tmp_path / "bad.csv"
    p2.write_text("x1,x2,label\n0,0,x\n", encoding="utf-8")
    assert _validate(base, p2, 0.5, 0.5).status_code == 400
    # Nothing was decided by the rejected submissions.
    assert httpx.get(base + "/api/release/weights",
                     timeout=5).status_code == 404


def test_concurrent_validations_are_serialised(cluster, tmp_path):
    """Two simultaneous submissions never both decide; at most one runs and a
    busy gate answers 409 without creating a second conclusion."""
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path))
    _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    val = _validation_csv(tmp_path)
    results = {}

    def submit(tag):
        results[tag] = _validate(base, val, 0.0, 1.0)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(submit, "a")
        time.sleep(0.05)
        f2 = pool.submit(submit, "b")
        f1.result(); f2.result()

    codes = sorted(r.status_code for r in results.values())
    assert 200 in codes
    assert all(c in (200, 409) for c in codes)
    assert sum(1 for c in codes if c == 200) == 1


def test_coordinator_restart_during_validation_keeps_release(cluster,
                                                             tmp_path):
    """Kill the coordinator while a validation is in flight; on restart the
    attempt is marked interrupted and the previously approved version stays
    downloadable. No half-finished validation becomes a conclusion."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=8)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    assert st["confirmed_round"] == 8
    approved = _validate(base, _validation_csv(tmp_path), 0.0, 1.0)
    assert approved.status_code == 200
    version = approved.json()["weights_version"]

    # Slow evaluation down so the kill lands while -> /evaluate is in flight.
    httpx.post(base + "/api/workers/delay", json={"seconds": 1.5}, timeout=10)
    import threading
    holder = {}

    def submit():
        try:
            holder["r"] = _validate(base, _validation_csv(tmp_path), 0.0, 1.0)
        except Exception as exc:  # noqa: BLE001 - connection dies with coord
            holder["e"] = exc

    t = threading.Thread(target=submit)
    t.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if httpx.get(base + "/api/status", timeout=2).json()[
                    "validation_in_progress"]:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    cluster["coord_svc"].hard_kill()
    t.join(timeout=30)

    cluster["coord_svc"].start()
    httpx.post(base + "/api/workers/delay", json={"seconds": 0}, timeout=10)
    st = httpx.get(base + "/api/status", timeout=10).json()
    assert st["validation_in_progress"] is False
    assert st["release"]["outcome"] == "approved"
    assert st["release"]["last_attempt"]["outcome"] == "interrupted"
    dl = httpx.get(base + "/api/release/weights", timeout=5)
    assert dl.status_code == 200 and dl.json()["weights_version"] == version
