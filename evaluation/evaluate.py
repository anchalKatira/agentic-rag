"""Evaluate the pipeline on a PDF + a small Q/A dataset.

Retrieval-only (no API key, tests ChromaDB + embeddings):
    python -m evaluation.evaluate --pdf paper.pdf --dataset evaluation/sample_dataset.json

Full agent run (needs GROQ_API_KEY; adds grader/retry/fallback/latency stats):
    python -m evaluation.evaluate --pdf paper.pdf --dataset my_qa.json --agent --out report.json

Dataset format (JSON list):
    [{"question": "...", "expected_keywords": ["..."], "expected_pages": [3]}]
`expected_keywords` / `expected_pages` are both optional; a hit needs either to match.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from core.agent import build_graph, run_agent
from core.config import TOP_K, get_api_key
from core.ingestion import ingest_pdf
from db.chroma_store import ChromaStore
from evaluation.metrics import answer_keyword_hit, mean, reciprocal_rank, retrieval_hit


def evaluate_retrieval(vectorstore, dataset, k: int = TOP_K) -> dict:
    rows = []
    for item in dataset:
        docs = vectorstore.similarity_search(item["question"], k=k)
        kw, pg = item.get("expected_keywords"), item.get("expected_pages")
        rows.append({
            "question": item["question"],
            "hit": retrieval_hit(docs, kw, pg),
            "rr": reciprocal_rank(docs, kw, pg),
        })
    return {
        "k": k,
        "n": len(rows),
        f"hit_rate@{k}": mean([float(r["hit"]) for r in rows]),
        "mrr": mean([r["rr"] for r in rows]),
        "rows": rows,
    }


def evaluate_agent(graph, dataset) -> dict:
    rows = []
    for item in dataset:
        t0 = time.perf_counter()
        result = run_agent(graph, item["question"], thread_id=str(uuid.uuid4()))
        latency = time.perf_counter() - t0
        steps = result.get("steps", [])
        rows.append({
            "question": item["question"],
            "answer_hit": answer_keyword_hit(result["answer"], item.get("expected_keywords")),
            "retried": any(s.startswith("Rephrase") for s in steps),
            "fallback": any(s.startswith("Fallback") for s in steps),
            "steps": len(steps),
            "latency_s": round(latency, 2),
        })
    scored = [float(r["answer_hit"]) for r in rows if r["answer_hit"] is not None]
    return {
        "n": len(rows),
        "answer_keyword_hit_rate": mean(scored),
        "retry_rate": mean([float(r["retried"]) for r in rows]),
        "fallback_rate": mean([float(r["fallback"]) for r in rows]),
        "avg_steps": mean([r["steps"] for r in rows]),
        "avg_latency_s": mean([r["latency_s"] for r in rows]),
        "rows": rows,
    }


def _print_summary(title: str, report: dict) -> None:
    print(f"\n== {title} ==")
    for key, val in report.items():
        if key == "rows":
            continue
        print(f"  {key:<26} {val:.3f}" if isinstance(val, float) else f"  {key:<26} {val}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate Agentic RAG")
    p.add_argument("--pdf", required=True, help="PDF to ingest (reused from ChromaDB if seen before)")
    p.add_argument("--dataset", required=True, help="JSON list of eval questions")
    p.add_argument("--k", type=int, default=TOP_K)
    p.add_argument("--agent", action="store_true", help="also run the full LangGraph agent (needs GROQ_API_KEY)")
    p.add_argument("--out", help="write full JSON report here")
    args = p.parse_args(argv)

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    pdf_path = Path(args.pdf)

    ing = ingest_pdf(pdf_path.read_bytes(), pdf_path.name, ChromaStore())
    print(f"Ingested {ing.filename}: {ing.pages} pages, {ing.chunks} chunks "
          f"({'reused from disk' if ing.reused else 'embedded fresh'})")

    report = {"retrieval": evaluate_retrieval(ing.vectorstore, dataset, args.k)}
    _print_summary("Retrieval", report["retrieval"])

    if args.agent:
        key = get_api_key()
        if not key:
            print("GROQ_API_KEY not set — cannot run --agent", file=sys.stderr)
            return 1
        report["agent"] = evaluate_agent(build_graph(ing.vectorstore, key), dataset)
        _print_summary("Agent", report["agent"])

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
