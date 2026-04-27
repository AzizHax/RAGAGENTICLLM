"""
phenorag/monitor_server.py

WebSocket server that wraps the Orchestrator and emits
live events to the React dashboard.

Usage:
    python -m phenorag.monitor_server --arch B2 --corpus data/test_corpus/ehr_ioa_test_50p.json --skip-extraction

Then open the dashboard in a browser → it connects to ws://localhost:8765
"""
from __future__ import annotations

import asyncio
import json
import time
import threading
import argparse
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict
from typing import Any, Dict, List, Set

import websockets
from websockets.server import serve

# ════════════════════════════════════════════════════════════════
# EVENT EMITTER (thread-safe queue → WebSocket broadcast)
# ════════════════════════════════════════════════════════════════

class EventEmitter:
    """Thread-safe event emitter that broadcasts to all connected WebSocket clients."""

    def __init__(self):
        self._clients: Set = set()
        self._loop: asyncio.AbstractEventLoop = None
        self._queue: asyncio.Queue = None

    def set_loop(self, loop):
        self._loop = loop
        self._queue = asyncio.Queue()

    def add_client(self, ws):
        self._clients.add(ws)

    def remove_client(self, ws):
        self._clients.discard(ws)

    def emit(self, agent: str, msg_type: str, text: str,
             probs: Dict[str, float] = None, meta: Dict = None):
        """Called from any thread — puts event on the async queue."""
        event = {
            "agent": agent,
            "type": msg_type,
            "text": text,
            "ts": time.time(),
        }
        if probs:
            event["probs"] = probs
        if meta:
            event["meta"] = meta

        if self._loop and self._queue:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    async def broadcaster(self):
        """Async loop that reads from queue and sends to all clients."""
        while True:
            event = await self._queue.get()
            msg = json.dumps(event, ensure_ascii=False)
            dead = set()
            for ws in self._clients:
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            self._clients -= dead


emitter = EventEmitter()


# ════════════════════════════════════════════════════════════════
# INSTRUMENTED ORCHESTRATOR
# ════════════════════════════════════════════════════════════════

def run_instrumented_pipeline(corpus_path: str, output_dir: str,
                               arch: str, skip_extraction: bool):
    """Run the real pipeline with emit() calls at each step."""
    from phenorag.agents.orchestrator import Orchestrator
    from phenorag.agents.configs import (
        Agent1Config, Agent2Config, Agent3Config, PipelineConfig)
    from phenorag.utils.llm_client import LLMClient
    from phenorag.utils.prompt_loader import PromptLoader

    emitter.emit("System", "info", f"Initializing PhenoRAG ({arch})...")

    llm = LLMClient(base_url="http://localhost:11434")

    prompts = PromptLoader(prompts_root="prompts")

    pcfg = PipelineConfig(architecture=arch)
    a1 = Agent1Config()
    a2 = Agent2Config()
    a3 = Agent3Config()

    orch = Orchestrator(llm=llm, prompts=prompts, pipeline_cfg=pcfg,
                        a1_cfg=a1, a2_cfg=a2, a3_cfg=a3,
                        kb_path="data/kb_pr_phenotype.json")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    facts_path = out / "facts.jsonl"
    assess_path = out / "assessments.jsonl"
    decisions_path = out / "decisions.jsonl"

    # Stage 1: Extraction
    if skip_extraction and facts_path.exists():
        emitter.emit("Agent1", "info", f"Skipping extraction, using {facts_path}")
    else:
        emitter.emit("Agent1", "extract", "Starting corpus extraction...")
        orch.agent1.process_corpus(corpus_path, str(facts_path))
        emitter.emit("Agent1", "info", "Extraction complete")

    with open(facts_path) as f:
        all_facts = [json.loads(line) for line in f]
    raw_corpus = json.load(open(corpus_path))
    raw_stays = {s["stay_id"]: s for p in raw_corpus for s in p.get("stays", [])}

    emitter.emit("System", "info", f"Processing {len(all_facts)} stays with {arch}")

    # Stage 2: Stay reasoning through the graph
    assessments = []
    for i, facts in enumerate(all_facts, 1):
        sid = facts.get("stay_id", "?")
        pid = facts.get("patient_id", "?")

        emitter.emit("Agent1", "extract",
                     f"[{i}/{len(all_facts)}] {pid}/{sid}: "
                     f"{len(facts.get('disease_mentions',[]))} diseases, "
                     f"{len(facts.get('labs',[]))} labs, "
                     f"{len(facts.get('drugs',[]))} drugs")

        # Build state and run graph
        state = {
            "stay_facts": facts, "raw_stay": raw_stays.get(sid),
            "patient_id": pid, "stay_id": sid,
            "assessment": None, "feedback_cycle": 0,
            "confirmed_absent": [], "route": "",
            "voter_assessments": [], "final_assessment": None, "trace": [],
        }
        config = {"configurable": {"thread_id": f"{pid}_{sid}"}}
        result = orch.graph.invoke(state, config)

        final = result.get("final_assessment", {})
        if final:
            assessments.append(final)
            label = final.get("final_stay_label", "?")
            conf = final.get("confidence", 0)
            comp = final.get("completeness", 0)
            probs = final.get("class_probabilities", {})
            pred_class = final.get("predicted_class", "")

            emitter.emit("Agent2", "reason",
                         f"{pid}/{sid}: ACR={final.get('acr_eular_final_score',0)}/10 "
                         f"→ {label} (conf={conf:.2f}, comp={comp:.2f})")

            if probs:
                pr_pos = final.get("pr_positive_probability", 0)
                emitter.emit("Probabilistic", "classify",
                             f"P(PR+)={pr_pos:.3f} → {pred_class}", probs=probs)

            # Emit trace messages
            for t in result.get("trace", []):
                if "Feedback" in t:
                    emitter.emit("Agent1", "feedback", t)
                elif "Critic" in t:
                    emitter.emit("Critic", "review", t)
                elif "Router" in t:
                    emitter.emit("Router", "route", t)
                elif "B4" in t:
                    emitter.emit("Agent2", "vote", t)

    with open(assess_path, "w") as f:
        for a in assessments:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    emitter.emit("System", "info", f"{len(assessments)} assessments saved")

    # Stage 3: Patient aggregation
    by_patient = defaultdict(list)
    for a in assessments:
        by_patient[a["patient_id"]].append(a)

    decisions = []
    for pid, pa in by_patient.items():
        dec = orch.agent3.process_patient(pa)
        dec_dict = asdict(dec)
        decisions.append(dec_dict)

        patient_probs = dec_dict.get("patient_class_probabilities", {})
        emitter.emit("Agent3", "aggregate",
                     f"{pid} ({dec.n_stays} stays): "
                     f"{dec.n_ra_positive_stays} RA+, {dec.n_ra_negative_stays} RA- "
                     f"→ {dec.final_label} (conf={dec.final_confidence:.2f})",
                     probs=patient_probs)

    with open(decisions_path, "w") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_pos = sum(1 for d in decisions if d["final_label"] == "RA+")
    emitter.emit("System", "done",
                 f"Pipeline complete: {len(decisions)} patients, "
                 f"{n_pos} RA+, {len(decisions)-n_pos} RA-",
                 meta={"n_patients": len(decisions), "n_ra_pos": n_pos,
                        "n_ra_neg": len(decisions)-n_pos})


# ════════════════════════════════════════════════════════════════
# WEBSOCKET SERVER
# ════════════════════════════════════════════════════════════════

async def ws_handler(websocket):
    emitter.add_client(websocket)
    print(f"  Client connected ({len(emitter._clients)} total)")
    try:
        async for msg in websocket:
            # Client can send commands
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "ping":
                    await websocket.send(json.dumps({"agent": "System", "type": "pong", "text": "connected"}))
            except Exception:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        emitter.remove_client(websocket)
        print(f"  Client disconnected ({len(emitter._clients)} total)")


async def main(args):
    emitter.set_loop(asyncio.get_event_loop())

    # Start broadcaster task
    asyncio.create_task(emitter.broadcaster())

    # Start WebSocket server
    print(f"\n{'='*60}")
    print(f"PhenoRAG Monitor Server")
    print(f"{'='*60}")
    print(f"  WebSocket: ws://localhost:{args.port}")
    print(f"  Architecture: {args.arch}")
    print(f"  Corpus: {args.corpus}")
    print(f"  Waiting for client connection...")

    async with serve(ws_handler, "0.0.0.0", args.port):
        # Wait for at least one client before starting pipeline
        while not emitter._clients:
            await asyncio.sleep(0.5)

        print(f"\n  Client connected, starting pipeline...")
        await asyncio.sleep(1)

        # Run pipeline in a thread (blocking LLM calls)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            run_instrumented_pipeline,
            args.corpus, args.output, args.arch, args.skip_extraction
        )

        # Keep server alive after pipeline finishes
        print(f"\n  Pipeline finished. Server stays alive for inspection.")
        print(f"  Press Ctrl+C to stop.")
        await asyncio.Future()  # block forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhenoRAG Live Monitor Server")
    parser.add_argument("--arch", default="B2", choices=["B1","B2","B3","B4"])
    parser.add_argument("--corpus", default="data/test_corpus/ehr_ioa_test_50p.json")
    parser.add_argument("--output", default="runs/live")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--skip-extraction", action="store_true")
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nServer stopped.")
