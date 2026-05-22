"""
Evaluation runner — Stage 4.

Usage:
    python -m evaluation.runner \\
        --dataset evaluation/dataset/eval_questions.json \\
        --output-dir evaluation/results/ \\
        [--max-questions N] \\
        [--dry-run]

  --dry-run  skips the live agent calls and writes placeholder results (for CI).
"""
import sys
import os
import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import Optional

# Ensure project root is on path when run as __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.memory.db import init_db
from app.memory.repositories import SessionRepository, ResearchTurnRepository
from app.research.agent import ResearchAgent


def _load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_result(out_file, result: dict):
    out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    out_file.flush()


async def _run_question(agent: ResearchAgent, question: dict, session_id: int, dry_run: bool) -> dict:
    """Run a single evaluation question and return the result dict."""
    q_id = question["id"]
    category = question["category"]
    q_text = question["question"]

    if dry_run:
        return {
            "id": q_id,
            "category": category,
            "question": q_text,
            "answer": f"[DRY RUN] Skipped agent call for question {q_id}.",
            "citations": [],
            "retrieved_chunks": [],
            "plan_text": "Dry run — no plan generated.",
            "search_queries": [],
            "latency_ms": 0.0,
            "error": None,
        }

    start = time.monotonic()
    try:
        result = await agent.run(q_text, session_id)
        latency_ms = round((time.monotonic() - start) * 1000, 1)

        plan_text = ""
        search_queries = []
        if result.plan:
            plan_text = result.plan.plan_text
            search_queries = result.plan.search_queries

        return {
            "id": q_id,
            "category": category,
            "question": q_text,
            "answer": result.answer or "",
            "citations": [c.model_dump() for c in (result.citations or [])],
            "retrieved_chunks": [c.model_dump() for c in (result.retrieved_chunks or [])],
            "plan_text": plan_text,
            "search_queries": search_queries,
            "latency_ms": latency_ms,
            "error": result.error or None,
        }
    except Exception as e:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        return {
            "id": q_id,
            "category": category,
            "question": q_text,
            "answer": "",
            "citations": [],
            "retrieved_chunks": [],
            "plan_text": "",
            "search_queries": [],
            "latency_ms": latency_ms,
            "error": str(e),
        }


async def run_evaluation(
    dataset_path: str,
    output_dir: str,
    max_questions: Optional[int] = None,
    dry_run: bool = False,
):
    init_db()
    session_repo = SessionRepository()

    questions = _load_dataset(dataset_path)
    if max_questions:
        questions = questions[:max_questions]

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "results.jsonl")

    print(f"[Runner] Evaluating {len(questions)} question(s).")
    print(f"[Runner] Results will be saved to: {output_path}")
    if dry_run:
        print("[Runner] DRY RUN — agent calls will be skipped.")

    # Separate conversational follow-ups into a shared session
    conv_session = session_repo.create(title="Eval: Conversational")
    main_session = session_repo.create(title="Eval: Main")

    agent = ResearchAgent()

    with open(output_path, "w", encoding="utf-8") as out_file:
        for i, question in enumerate(questions, 1):
            category = question.get("category", "")
            session_id = conv_session.id if category == "conversational_followup" else main_session.id

            print(f"[Runner] [{i}/{len(questions)}] {question['id']} — {question['question'][:60]}...")
            result = await _run_question(agent, question, session_id, dry_run)

            status = "ERROR" if result.get("error") else "OK"
            print(f"[Runner]   → {status} | {result['latency_ms']:.0f}ms | "
                  f"{len(result['citations'])} citation(s)")

            _write_result(out_file, result)

    print(f"\n[Runner] Done. Results written to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Deep Research Evaluation Runner")
    parser.add_argument("--dataset", required=True, help="Path to eval_questions.json")
    parser.add_argument("--output-dir", default="evaluation/results", help="Directory for output files")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions")
    parser.add_argument("--dry-run", action="store_true", help="Skip live agent calls (for CI)")
    args = parser.parse_args()

    asyncio.run(run_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_questions=args.max_questions,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
