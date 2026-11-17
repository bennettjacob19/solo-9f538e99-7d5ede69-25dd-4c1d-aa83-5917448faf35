"""Release-gate math and integrity checks.

The production validation flow freezes one confirmed training version
``(task_id, confirmed_round, weights)`` and asks both workers to classify a
mutually-exclusive subset of a validation CSV *with exactly those weights*.
This module contains the pure pieces shared by the coordinator and workers:

* a canonical, content-addressed version string for the frozen weights
  (``weights_version``);
* confusion-count computation at a fixed decision threshold;
* merging of per-worker counts and derivation of recall / false-positive rate;
* the pass/reject decision against the operator's thresholds;
* strict verification of worker replies (task, version, sample count and
  counts) so a stale/mixed reply can never be turned into a new conclusion.

Nothing here reads training state. A worker may have durably applied an
unconfirmed round; evaluation still uses only the weights supplied in the
request, which is what makes the frozen snapshot meaningful.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence

import numpy as np

import lrmath

# Fixed classifier operating point. Thresholds supplied by the operator are
# acceptance criteria for the model version, not a per-sample score cut-off.
DECISION_THRESHOLD: float = 0.5
COUNT_KEYS = ("tp", "fp", "tn", "fn")


class GateError(Exception):
    """A validation reply is unusable; no new release conclusion may be made."""


def normalize_weights(weights: Sequence[float]) -> list[float]:
    """Validate and canonicalise the frozen weight vector (bias, w1, w2)."""
    try:
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise GateError(f"weights are not numeric: {exc}") from exc
    if arr.shape != (lrmath.N_WEIGHTS,):
        raise GateError(
            f"weights must contain exactly {lrmath.N_WEIGHTS} values")
    if not np.all(np.isfinite(arr)):
        raise GateError("weights must be finite")
    return [float(x) for x in arr]


def weights_version(task_id: str, round_: int, weights: Sequence[float]) -> str:
    """Content-addressed version of a frozen model snapshot.

    Includes the task id and confirmed round, so a version from another task or
    another round can never compare equal even if the numbers happen to match.
    """
    values = normalize_weights(weights)
    blob = json.dumps(
        {"task_id": str(task_id), "round": int(round_), "w": values},
        separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"r{int(round_)}:{hashlib.sha256(blob).hexdigest()}"


def normalize_threshold(value: float) -> float:
    """Validate an acceptance threshold in the closed interval [0, 1]."""
    try:
        val = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"threshold is not numeric: {exc}") from exc
    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
        raise GateError("thresholds must be finite and within [0, 1]")
    return val


def _check_threshold(value: float) -> float:
    return normalize_threshold(value)


def normalize_counts(counts: Mapping[str, Any]) -> Dict[str, int]:
    """Return exactly the four non-negative integer confusion counts."""
    if not isinstance(counts, Mapping):
        raise GateError("confusion counts are missing")
    out: Dict[str, int] = {}
    for key in COUNT_KEYS:
        if key not in counts:
            raise GateError(f"confusion counts miss '{key}'")
        value = counts[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise GateError(f"confusion count '{key}' is not an integer")
        if value < 0:
            raise GateError(f"confusion count '{key}' is negative")
        out[key] = value
    return out


def confusion_counts(x_aug: np.ndarray, y: Sequence[float],
                     weights: Sequence[float],
                     threshold: float = DECISION_THRESHOLD) -> Dict[str, int]:
    """Confusion counts for a set of samples at ``threshold``.

    Positive prediction is ``p >= threshold`` (fixed once for every version, so
    recall/FPR comparisons across candidates are like-for-like).
    """
    cut = _check_threshold(threshold)
    w = np.asarray(normalize_weights(weights), dtype=np.float64)
    x = np.asarray(x_aug, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != labels.shape[0] or x.shape[1] != 3:
        raise GateError("evaluation matrix and labels have incompatible shapes")
    if labels.size == 0:
        raise GateError("evaluation set is empty")
    if not np.all((labels == 0.0) | (labels == 1.0)):
        raise GateError("validation labels must be 0 or 1")

    pred = (lrmath.probabilities(x, w) >= cut)
    pos = labels == 1.0
    return {
        "tp": int(np.count_nonzero(pred & pos)),
        "fp": int(np.count_nonzero(pred & ~pos)),
        "tn": int(np.count_nonzero(~pred & ~pos)),
        "fn": int(np.count_nonzero(~pred & pos)),
    }


def merge_counts(parts: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Sum per-worker counts. The caller must already have verified them."""
    if not parts:
        raise GateError("no worker counts to merge")
    total = {key: 0 for key in COUNT_KEYS}
    for part in parts:
        counts = normalize_counts(part)
        for key in COUNT_KEYS:
            total[key] += counts[key]
    return total


def metrics_from_counts(counts: Mapping[str, Any]) -> Dict[str, Any]:
    """Recall, false-positive rate and supporting metrics over merged counts."""
    c = normalize_counts(counts)
    n = sum(c.values())
    positives = c["tp"] + c["fn"]
    negatives = c["fp"] + c["tn"]
    if n == 0:
        raise GateError("merged validation set is empty")
    if positives == 0:
        raise GateError("validation set contains no positive samples")
    if negatives == 0:
        raise GateError("validation set contains no negative samples")
    predicted_positive = c["tp"] + c["fp"]
    return {
        "n": n,
        "positives": positives,
        "negatives": negatives,
        "recall": c["tp"] / positives,
        "false_positive_rate": c["fp"] / negatives,
        "accuracy": (c["tp"] + c["tn"]) / n,
        "precision": (c["tp"] / predicted_positive
                      if predicted_positive else None),
    }


def decide(counts: Mapping[str, Any], min_recall: float,
           max_fpr: float) -> Dict[str, Any]:
    """Apply the operator's acceptance thresholds to merged confusion counts."""
    min_floor = _check_threshold(min_recall)
    max_ceiling = _check_threshold(max_fpr)
    metrics = metrics_from_counts(counts)
    problems: list[str] = []
    if metrics["recall"] < min_floor:
        problems.append(
            f"召回率 {metrics['recall']:.6f} 低于最低要求 {min_floor:.6f}")
    if metrics["false_positive_rate"] > max_ceiling:
        problems.append(
            f"误报率 {metrics['false_positive_rate']:.6f} 高于最高限制 "
            f"{max_ceiling:.6f}")
    passed = not problems
    return {
        "passed": passed,
        "reason": "满足放行门槛" if passed else "；".join(problems),
        "metrics": metrics,
    }


def verify_shard_response(response: Any, *, task_id: str, validation_id: str,
                          version: str, worker_id: str, round_: int,
                          expected_n: int,
                          expected_counts: Mapping[str, Any]) -> Dict[str, int]:
    """Reject any worker reply that does not exactly describe the frozen run.

    Checks task id, validation id, version, worker identity, round, sample
    count and the worker's confusion counts against a coordinator-side
    recomputation from the frozen weights. A mismatch raises :class:`GateError`
    and therefore produces no new conclusion (the previous release is kept).
    """
    if not isinstance(response, Mapping):
        raise GateError(f"{worker_id}: malformed evaluation reply")
    if "_error" in response:
        raise GateError(f"{worker_id}: {response['_error']}")
    if response.get("task_id") != task_id:
        raise GateError(f"{worker_id}: task id mismatch in evaluation reply")
    if response.get("validation_id") != validation_id:
        raise GateError(f"{worker_id}: validation id mismatch")
    if response.get("version") != version:
        raise GateError(f"{worker_id}: frozen weights version mismatch")
    if response.get("worker_id") != worker_id:
        raise GateError(f"{worker_id}: reply came from another worker")
    try:
        echoed_round = int(response.get("round"))
        echoed_n = int(response.get("n"))
    except (TypeError, ValueError) as exc:
        raise GateError(f"{worker_id}: malformed round/sample count") from exc
    if echoed_round != int(round_):
        raise GateError(f"{worker_id}: round mismatch in evaluation reply")
    if echoed_n != int(expected_n):
        raise GateError(
            f"{worker_id}: sample count mismatch (expected {expected_n}, "
            f"got {echoed_n})")
    counts = normalize_counts(response.get("counts"))
    if counts != dict(expected_counts):
        raise GateError(
            f"{worker_id}: confusion counts do not match the frozen-weight "
            "recomputation")
    if sum(counts.values()) != int(expected_n):
        raise GateError(f"{worker_id}: confusion counts do not cover samples")
    return counts
