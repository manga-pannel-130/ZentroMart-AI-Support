"""Compare the same LLM in Step 1 (prompt-only) and Step 2 (RAG)."""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from zentro_agent import DATA_ROOT, ROOT, HybridRetriever, LocalGenerator, ZentroSupportAgent, find_product, load_sections


def normalize(value: str) -> str:
    digits = []
    for char in value.casefold():
        try:
            digits.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            digits.append(char)
    return " ".join(re.sub(r"[^\w₹]+", " ", "".join(digits), flags=re.UNICODE).split())


def fact_groups(cell: str) -> list[list[str]]:
    return [[part.strip() for part in group.split("/") if part.strip()]
            for group in (cell or "").split(";") if group.strip()]


def score(answer: str, row: dict[str, str]) -> dict[str, bool]:
    normalized = normalize(answer)
    required = all(any(normalize(option) in normalized for option in group)
                   for group in fact_groups(row.get("must_include", "")))
    forbidden = [x.strip() for x in (row.get("must_not_include") or "").split(";") if x.strip()]
    no_forbidden = not any(normalize(term) in normalized for term in forbidden if normalize(term))
    return {"required_facts_pass": required, "forbidden_phrases_pass": no_forbidden,
            "strict_heuristic_pass": required and no_forbidden}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "step_comparison.json")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    from sentence_transformers import SentenceTransformer
    questions = list(csv.DictReader((DATA_ROOT / "practice" / "questions.csv").open(
        encoding="utf-8-sig", newline="")))
    embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    retriever = HybridRetriever(load_sections(), embedder, top_k=args.top_k)
    model = LocalGenerator(args.model_id)
    agent = ZentroSupportAgent(retriever, model, mock_tools=False)
    prompt_only_system = (
        "You are ZentroMart's AI customer-support assistant. Reply in the customer's language in 2-3 short sentences. "
        "You have not been given company documents or customer/order data. Do not pretend you looked anything up. "
        "If you cannot answer reliably, say you do not know and offer a human colleague. Never invent policies, fees, dates, or actions.")
    rows = []
    for i, row in enumerate(questions, start=1):
        question = row["question"]
        baseline = model.generate(prompt_only_system, f"Customer question: {question}")
        result = agent.respond(question, {})
        rag = result["text"]
        predicted_sources = result.get("evidence", [])
        rows.append({"id": row["id"], "language": row["language"], "question": question,
                     "answerable": row["answerable"], "gold_evidence": row["gold_evidence"],
                     "step1_prompt_only": {"answer": baseline, **score(baseline, row)},
                     "step2_rag": {"answer": rag, "retrieved_sections": predicted_sources,
                                   "decision": result.get("decision"),
                                   **score(rag, row)}})
        print(f"[{i}/{len(questions)}] {row['id']}")

    def rate(step: str) -> dict[str, float]:
        return {metric: sum(bool(r[step][metric]) for r in rows) / max(1, len(rows))
                for metric in ("required_facts_pass", "forbidden_phrases_pass", "strict_heuristic_pass")}

    evidence_rows = [row for row in rows if any(ref.strip() for ref in row["gold_evidence"].split(";"))]
    evidence_hits = []
    reciprocal_ranks = []
    for row in evidence_rows:
        gold = {ref.strip() for ref in row["gold_evidence"].split(";") if ref.strip()}
        ranked = row["step2_rag"]["retrieved_sections"]
        matches = [rank + 1 for rank, ref in enumerate(ranked) if ref in gold]
        evidence_hits.append(bool(matches))
        reciprocal_ranks.append(1 / min(matches) if matches else 0)

    report = {"dataset": "practice/questions.csv (visible development set; not held out)",
              "model_id": model.model_id, "question_count": len(rows), "top_k": args.top_k,
              "step1_prompt_only": rate("step1_prompt_only"), "step2_rag": rate("step2_rag"),
              "rag_retrieval": {
                  "evidence_eligible_questions": len(evidence_rows),
                  "gold_hit_at_k": sum(evidence_hits) / max(1, len(evidence_rows)),
                  "mrr_at_k": sum(reciprocal_ranks) / max(1, len(evidence_rows)),
                  "note": "Gold hit is evidence recall; strict answer pass uses literal phrase checks and is not semantic grading."},
              "limitations": ["Visible practice examples are development data, not hidden evaluation.",
                              "String/phrase matching is heuristic and can reject correct paraphrases or accept unsupported answers.",
                              "Review every answer and retrieval miss before reporting quality."]}
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": report, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
