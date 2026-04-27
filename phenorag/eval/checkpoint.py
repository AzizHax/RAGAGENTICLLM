"""
phenorag/eval/checkpoint.py

Système de checkpoint temporel pour les longs runs.

Idée :
  - Un checkpoint manager tourne en parallèle du pipeline.
  - À intervalles réguliers (default 5 min), il flush les décisions déjà
    écrites dans decisions.jsonl vers un checkpoint.json (atomique).
  - Au démarrage, le runner lit le checkpoint et skip les patients déjà traités.

Usage côté pipeline :
    cm = CheckpointManager(run_dir, interval_min=5)
    cm.start()
    for patient in patients:
        if cm.is_done(patient_id):
            continue          # déjà traité
        ...
        cm.mark_done(patient_id)
    cm.stop()
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Set


class CheckpointManager:
    """Sauvegarde périodique des patients traités + reprise auto."""

    def __init__(self, run_dir: str | Path, interval_min: float = 5.0):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.interval_s = interval_min * 60
        self._done: Set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    # ── Persistence ──────────────────────────────────────────────

    def _load(self):
        """Recharge un checkpoint existant si présent."""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._done = set(data.get("done", []))
                print(f"  [Checkpoint] Reprise: {len(self._done)} patients déjà traités")
            except Exception as e:                                         # noqa: BLE001
                print(f"  [Checkpoint] Lecture échouée ({e}), repart de zéro")
                self._done = set()

    def _flush(self):
        """Écriture atomique du checkpoint (write tmp + rename)."""
        with self._lock:
            data = {"done": sorted(self._done), "n": len(self._done),
                    "timestamp": time.time()}
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(self.checkpoint_path)

    # ── API ──────────────────────────────────────────────────────

    def is_done(self, patient_id: str) -> bool:
        with self._lock:
            return str(patient_id) in self._done

    def mark_done(self, patient_id: str):
        with self._lock:
            self._done.add(str(patient_id))

    def n_done(self) -> int:
        with self._lock:
            return len(self._done)

    # ── Thread de sauvegarde ─────────────────────────────────────

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            self._stop.wait(self.interval_s)
            try:
                self._flush()
            except Exception as e:                                         # noqa: BLE001
                print(f"  [Checkpoint] flush échoué: {e}")

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=5.0)
        self._flush()   # flush final
        print(f"  [Checkpoint] Final: {self.n_done()} patients sauvegardés")
