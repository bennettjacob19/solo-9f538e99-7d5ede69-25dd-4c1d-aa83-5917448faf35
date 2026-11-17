"""Unit tests for the release-gate math, integrity checks and release store."""

import json

import numpy as np
import pytest

import gate
import lrmath
from release import ReleaseStore


def _toy(n=20, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.float64)
    return x, y


def test_weights_version_is_content_addressed():
    w = [0.1, -0.2, 0.3]
    v = gate.weights_version("task-a", 4, w)
    assert v == gate.weights_version("task-a", 4, list(w))
    assert v.startswith("r4:")
    assert v != gate.weights_version("task-b", 4, w)
    assert v != gate.weights_version("task-a", 5, w)
    assert v != gate.weights_version("task-a", 4, [0.1, -0.2, 0.300001])


def test_weights_version_rejects_bad_shape_and_nonfinite():
    for bad in ([1.0, 2.0], [1, 2, 3, 4], [float("nan"), 0, 0],
                [float("inf"), 0, 0], ["a", "b", "c"]):
        with pytest.raises(gate.GateError):
            gate.weights_version("t", 0, bad)


def test_confusion_counts_uses_fixed_threshold():
    x = np.array([[0.0, 0.0], [10.0, 10.0], [-10.0, -10.0], [1.0, 0.0]])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    w = [0.0, 1.0, 1.0]  # p = sigmoid(x1+x2)
    counts = gate.confusion_counts(lrmath.augment(x), y, w, 0.5)
    # [0,0] -> p=0.5 (counts as positive, >= cut), [1,0] -> 0.731 (positive),
    # [-10,-10] -> ~0 (negative), [10,10] -> ~1 (positive).
    assert counts == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert sum(counts.values()) == len(y)


def test_merge_and_metrics_match_full_set():
    x, y = _toy(21)
    w = [0.05, -0.4, 0.8]
    cut = (len(y) + 1) // 2
    a = gate.confusion_counts(lrmath.augment(x[:cut]), y[:cut], w)
    b = gate.confusion_counts(lrmath.augment(x[cut:]), y[cut:], w)
    merged = gate.merge_counts([a, b])
    full = gate.confusion_counts(lrmath.augment(x), y, w)
    assert merged == full

    metrics = gate.metrics_from_counts(merged)
    assert metrics["n"] == len(y)
    assert metrics["recall"] == merged["tp"] / (merged["tp"] + merged["fn"])
    assert metrics["false_positive_rate"] == (
        merged["fp"] / (merged["fp"] + merged["tn"]))


def test_decide_threshold_boundaries_are_inclusive():
    counts = {"tp": 8, "fp": 1, "tn": 9, "fn": 2}  # recall .8, fpr .1
    assert gate.decide(counts, 0.8, 0.1)["passed"] is True
    assert gate.decide(counts, 0.81, 0.1)["passed"] is False
    assert gate.decide(counts, 0.8, 0.099)["passed"] is False


def test_decide_rejects_single_class_or_empty_validation():
    with pytest.raises(gate.GateError):
        gate.metrics_from_counts({"tp": 3, "fp": 0, "tn": 0, "fn": 1})
    with pytest.raises(gate.GateError):
        gate.metrics_from_counts({"tp": 0, "fp": 0, "tn": 0, "fn": 0})


def test_normalize_counts_strict():
    with pytest.raises(gate.GateError):
        gate.normalize_counts({"tp": 1, "fp": 0, "tn": 0})
    with pytest.raises(gate.GateError):
        gate.normalize_counts({"tp": True, "fp": 0, "tn": 0, "fn": 0})
    with pytest.raises(gate.GateError):
        gate.normalize_counts({"tp": 1.5, "fp": 0, "tn": 0, "fn": 0})
    with pytest.raises(gate.GateError):
        gate.normalize_counts({"tp": -1, "fp": 0, "tn": 0, "fn": 0})


def test_verify_shard_response_accepts_exact_and_rejects_mismatch():
    w = [0.2, 0.1, -0.3]
    task, vid = "task-1", "val-1"
    version = gate.weights_version(task, 7, w)
    expected = {"tp": 2, "fp": 1, "tn": 3, "fn": 0}
    reply = {"worker_id": "worker1", "task_id": task,
             "validation_id": vid, "version": version, "round": 7,
             "n": 6, "counts": expected}
    kwargs = dict(task_id=task, validation_id=vid, version=version,
                  worker_id="worker1", round_=7, expected_n=6,
                  expected_counts=expected)
    assert gate.verify_shard_response(reply, **kwargs) == expected

    mutations = [
        {"task_id": "other"},
        {"validation_id": "other"},
        {"version": gate.weights_version(task, 8, w)},
        {"worker_id": "worker2"},
        {"round": 6},
        {"n": 5},
        {"counts": {**expected, "tp": 3}},
        {"counts": {**expected, "fp": 2}},
        {"_error": "worker1: unreachable"},
    ]
    for change in mutations:
        with pytest.raises(gate.GateError):
            gate.verify_shard_response({**reply, **change}, **kwargs)


def test_release_store_publish_and_clear(tmp_path):
    store = ReleaseStore(str(tmp_path))
    task = "task-1"
    doc = store.new_doc(task, 1000)
    w = [0.5, -0.5, 0.25]
    doc = store.publish(
        doc, task_id=task, round_=9, weights=w,
        thresholds={"min_recall": 0.8, "max_fpr": 0.1},
        confusion={"tp": 8, "fp": 1, "tn": 9, "fn": 2},
        metrics=gate.metrics_from_counts(
            {"tp": 8, "fp": 1, "tn": 9, "fn": 2}),
        reason="ok", validation_id="v1", n=20, threshold=0.5, now_ms=2000)

    found = store.approved_artifact(task)
    assert found is not None
    payload = found["payload"]
    assert payload["weights_version"] == gate.weights_version(task, 9, w)
    assert payload["weights"] == {"bias": 0.5, "w1": -0.5, "w2": 0.25}
    assert store.summary(task)["download_available"] is True
    assert store.summary("other-task")["download_available"] is False

    # The record transitions to rejected while the approved artifact stays.
    doc = store.record_attempt(doc, {"validation_id": "v2",
                                     "outcome": "rejected"}, 3000)
    store.save(doc)
    assert store.summary(task)["outcome"] == "approved"
    assert store.summary(task)["last_attempt"]["validation_id"] == "v2"
    assert store.approved_artifact(task) is not None

    # Clearing removes the record and every released artifact.
    store.clear()
    assert store.load() is None
    assert store.approved_artifact(task) is None
    assert not (tmp_path / "releases").exists()


def test_release_store_rejects_tampered_or_missing_artifact(tmp_path):
    store = ReleaseStore(str(tmp_path))
    task = "task-1"
    doc = store.new_doc(task, 1000)
    w = [0.5, -0.5, 0.25]
    doc = store.publish(
        doc, task_id=task, round_=2, weights=w,
        thresholds={"min_recall": 0.5, "max_fpr": 0.5},
        confusion={"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        metrics=gate.metrics_from_counts(
            {"tp": 1, "fp": 1, "tn": 1, "fn": 1}),
        reason="ok", validation_id="v1", n=4, threshold=0.5, now_ms=2000)
    name = doc["conclusion"]["artifact"]
    path = tmp_path / "releases" / name

    payload = json.loads(path.read_text())
    payload["weights"]["w1"] = 0.9  # tamper with the served weights
    path.write_text(json.dumps(payload))
    assert store.approved_artifact(task) is None

    path.unlink()  # missing artifact must not be served either
    assert store.approved_artifact(task) is None


def test_mark_interrupted_leaves_conclusion_untouched(tmp_path):
    store = ReleaseStore(str(tmp_path))
    task = "task-1"
    doc = store.new_doc(task, 1000)
    doc = store.publish(
        doc, task_id=task, round_=3, weights=[0.1, 0.2, 0.3],
        thresholds={"min_recall": 0.5, "max_fpr": 0.5},
        confusion={"tp": 1, "fp": 0, "tn": 1, "fn": 0},
        metrics=gate.metrics_from_counts({"tp": 1, "fp": 0, "tn": 1, "fn": 0}),
        reason="ok", validation_id="v-ok", n=2, threshold=0.5, now_ms=2000)
    store.record_attempt(doc, {"validation_id": "v-open"}, 3000)
    store.save(doc)

    out = store.mark_interrupted(4000)
    assert out is not None
    assert out["conclusion"]["validation_id"] == "v-ok"
    assert out["conclusion"]["outcome"] == "approved"
    assert out["attempts"][-1]["outcome"] == "interrupted"
    assert store.approved_artifact(task) is not None
    # Nothing left to close on a second pass.
    assert store.mark_interrupted(5000) is None
