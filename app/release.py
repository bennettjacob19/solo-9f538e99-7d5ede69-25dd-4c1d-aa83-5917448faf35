"""Durable release-gate records.

The release record is deliberately kept *outside* the training status document:

* the training loop rewrites ``status.json`` every round, so putting release
  state there would clobber it;
* a released model version must survive later confirmed training rounds, and
  clearing the task must make it unavailable again.

Layout
------
``release.json``            current conclusion (or none) + recent attempts
``releases/<name>.json``    immutable, content-addressed weight artifact

Publishing writes the immutable artifact first and only then swaps the record,
so a crash or failed write leaves the previous conclusion (if any) intact.
Clearing deletes the whole ``releases/`` directory, which is why a cleared task
can no longer serve an old version.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Mapping, Optional

import gate
from storage import atomic_write_json, if_exists_read_json

MAX_ATTEMPTS = 25


class ReleaseStore:
    def __init__(self, data_dir: str) -> None:
        self.path = os.path.join(data_dir, "release.json")
        self.dir = os.path.join(data_dir, "releases")

    # ------------------------------------------------------------- record I/O
    def load(self) -> Optional[Dict[str, Any]]:
        return if_exists_read_json(self.path)

    def save(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        atomic_write_json(self.path, doc)
        return doc

    def clear(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def new_doc(self, task_id: str, now_ms: int) -> Dict[str, Any]:
        return {"task_id": task_id, "conclusion": None, "attempts": [],
                "updated_at": now_ms}

    # ---------------------------------------------------------------- attempts
    @staticmethod
    def record_attempt(doc: Dict[str, Any], attempt: Dict[str, Any],
                       now_ms: int) -> Dict[str, Any]:
        """Record a failed/incomplete validation.

        The frozen ``conclusion`` is intentionally left untouched, so a
        rejected or interrupted validation keeps the previously released
        version downloadable.
        """
        attempts: List[Dict[str, Any]] = list(doc.get("attempts") or [])
        attempts.append(attempt)
        doc["attempts"] = attempts[-MAX_ATTEMPTS:]
        doc["updated_at"] = now_ms
        return doc

    @staticmethod
    def update_attempt(doc: Dict[str, Any], validation_id: str,
                       changes: Mapping[str, Any], now_ms: int) -> Dict[str, Any]:
        for attempt in doc.get("attempts") or []:
            if attempt.get("validation_id") == validation_id:
                attempt.update(changes)
        doc["updated_at"] = now_ms
        return doc

    def mark_interrupted(self, now_ms: int) -> Optional[Dict[str, Any]]:
        """Close attempts left without an outcome by a coordinator crash.

        The conclusion is never touched, so an interrupted validation keeps the
        previously released version. Returns the saved doc, or ``None`` when
        there was nothing to close.
        """
        doc = self.load()
        if not doc:
            return None
        changed = False
        for attempt in doc.get("attempts") or []:
            if not attempt.get("outcome"):
                attempt["outcome"] = "interrupted"
                attempt["error"] = ("coordinator restarted before a "
                                    "conclusion was reached")
                attempt["finished_at"] = now_ms
                changed = True
        if changed:
            doc["updated_at"] = now_ms
            self.save(doc)
        return doc if changed else None

    # --------------------------------------------------------------- publish
    def publish(self, doc: Dict[str, Any], *, task_id: str, round_: int,
                weights: List[float], thresholds: Mapping[str, float],
                confusion: Mapping[str, int], metrics: Mapping[str, Any],
                reason: str, validation_id: str, n: int,
                threshold: float, now_ms: int) -> Dict[str, Any]:
        version = gate.weights_version(task_id, round_, weights)
        payload = {
            "kind": "defect_classifier_release_v1",
            "task_id": task_id,
            "round": int(round_),
            "weights_version": version,
            "weights": {"bias": weights[0], "w1": weights[1],
                        "w2": weights[2]},
            "decision_threshold": float(threshold),
            "thresholds": dict(thresholds),
            "confusion": dict(confusion),
            "metrics": dict(metrics),
            "outcome": "approved",
            "reason": reason,
            "validation_id": validation_id,
            "n": int(n),
            "decided_at": now_ms,
        }
        os.makedirs(self.dir, exist_ok=True)
        digest = version.split(":", 1)[1][:12]
        filename = f"release_round_{int(round_):04d}_{digest}.json"
        artifact_path = os.path.join(self.dir, filename)
        # Immutable artifact first; the record swap below is the commit point.
        # A crash/failed write leaves the previous approved release intact.
        atomic_write_json(artifact_path, payload)
        conclusion = dict(payload)
        conclusion["artifact"] = filename
        doc["task_id"] = task_id
        doc["conclusion"] = conclusion
        doc["updated_at"] = now_ms
        self.save(doc)
        return doc

    # -------------------------------------------------------------- download
    def approved_artifact(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Return ``{record, payload}`` only for an approved, intact release.

        The artifact is re-verified against its own weights so a truncated or
        tampered file can never be served as the released model.
        """
        doc = self.load()
        if not doc or doc.get("task_id") != task_id:
            return None
        record = doc.get("conclusion")
        if not record or record.get("outcome") != "approved":
            return None
        name = record.get("artifact")
        if not isinstance(name, str) or os.path.basename(name) != name:
            return None
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return None
        payload = if_exists_read_json(path)
        if not payload:
            return None
        try:
            w = payload["weights"]
            recomputed = gate.weights_version(
                payload["task_id"], payload["round"],
                [w["bias"], w["w1"], w["w2"]])
        except (KeyError, TypeError, ValueError, gate.GateError):
            return None
        if (payload.get("weights_version") != recomputed
                or recomputed != record.get("weights_version")
                or payload.get("task_id") != task_id
                or payload.get("outcome") != "approved"):
            return None
        return {"record": record, "payload": payload, "path": path}

    def summary(self, task_id: Optional[str]) -> Dict[str, Any]:
        """Public, non-sensitive view of the current release state."""
        doc = self.load()
        matches = bool(doc and task_id and doc.get("task_id") == task_id)
        conclusion = (doc or {}).get("conclusion") if matches else None
        attempts = ((doc or {}).get("attempts") or []) if matches else []
        last_attempt = attempts[-1] if attempts else None
        download = bool(task_id and self.approved_artifact(task_id))
        return {
            "task_matches": matches,
            "outcome": (conclusion or {}).get("outcome"),
            "version": (conclusion or {}).get("weights_version"),
            "round": (conclusion or {}).get("round"),
            "metrics": (conclusion or {}).get("metrics"),
            "confusion": (conclusion or {}).get("confusion"),
            "thresholds": (conclusion or {}).get("thresholds"),
            "reason": (conclusion or {}).get("reason"),
            "decided_at": (conclusion or {}).get("decided_at"),
            "download_available": download,
            "last_attempt": last_attempt,
        }
