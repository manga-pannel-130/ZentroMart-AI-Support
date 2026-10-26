"""Evaluate retrieval and (optionally) answer checks on supplied practice questions.

The supplied questions are development data, not a hidden or held-out benchmark.
Do not report their scores as final generalization performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

from zentro_agent import (ROOT, DATA_ROOT, HybridRetriever, LocalGenerator, ZentroSupportAgent,
                          build_product_index, find_product, load_sections)


def groups(cell: str) -> list[list[str]]:
    result = []
    for required in (cell or "").split(";"):
        alternatives = [x.strip() for x in required.split("/") if x.strip()]
        if alternatives:
            result.append(alternatives)
    return result


def normalize(value: str) -> str:
    chars = []
    for char in value.casefold():
        try:
            chars.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            chars.append(char)
    value = "".join(chars)
    value = re.sub(r"[^\w₹]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def contains_alternatives(answer: str, alternatives: list[str]) -> bool:
    normalized_answer = normalize(answer)
    return any(normalize(alt) in normalized_answer for alt in alternatives if normalize(alt))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true",
                        help="also run the local language model and score basic required/forbidden facts")
    parser.add_argument("--out", default="reports/practice_evaluation",
                        help="output file stem (creates .json and, with --generate, .csv)")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    embedder_id = __import__("os").getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-small")
    embedder = SentenceTransformer(embedder_id, device="cpu")
    retriever = HybridRetriever(load_sections(), embedder)
    products, sku_map = build_product_index()
    questions = list(csv.DictReader((DATA_ROOT / "practice" / "questions.csv").open(encoding="utf-8-sig", newline="")))
    generator = LocalGenerator() if args.generate else None
    agent = ZentroSupportAgent(retriever, generator, mock_tools=False) if generator else None

    rows = []
    retrieval_hits = retrieval_complete = mrr_total = 0
    answer_rows = []
    ans_correct = required_ok = forbidden_ok = citations_ok = 0
    generated_count = 0
    hard_negative_cases = 0
    hard_negative_eligible = 0
    for row in questions:
        question = row["question"]
        product = find_product(question, products, sku_map)
        search_query = question
        if product:
            search_query += f" return category {product['return_category']} warranty seller Assured refund eligibility"
        evidence = retriever.search(search_query, top_k=args.top_k)
        ranked_ids = [x["section_id"] for x in evidence]
        gold = {x.strip() for x in row["gold_evidence"].split(";") if x.strip()}
        predicted = set(ranked_ids)
        if product:
            predicted.add("products.csv:" + product["sku"])
        gold_kb = {x for x in gold if not x.startswith("products.csv:")}
        gold_products = {x for x in gold if x.startswith("products.csv:")}
        for ref in gold_products:
            sku = ref.split(":", 1)[1]
            if sku == (product or {}).get("sku"):
                predicted.add(ref)
        rank_candidates = []
        for ref in gold_kb:
            if ref in ranked_ids:
                rank_candidates.append(ranked_ids.index(ref) + 1)
        if product and "products.csv:" + product["sku"] in gold_products:
            rank_candidates.append(1)  # deterministic SKU/title table lookup
        has_gold = bool(gold_kb or gold_products)
        hit = bool(gold & predicted)
        if has_gold:
            retrieval_hits += hit
            retrieval_complete += bool(gold <= predicted)
            mrr_total += 1 / min(rank_candidates) if rank_candidates else 0
        hard = {x.strip() for x in row["hard_negatives"].split(";") if x.strip()}
        hard_negative_eligible += bool(hard)
        hard_hit = bool(hard & set(ranked_ids))
        hard_negative_cases += hard_hit
        record = {"id": row["id"], "language": row["language"], "question": question,
                  "answerable": row["answerable"], "expected_decision": row["expected_decision"],
                  "gold_evidence": sorted(gold), "retrieved_sections": ranked_ids,
                  "catalog_match": (product or {}).get("sku"), "evidence_hit_at_k": hit if has_gold else None,
                  "all_evidence_present": bool(gold <= predicted) if has_gold else None,
                  "hard_negative_retrieved": hard_hit, "answer": ""}

        if agent and row["answerable"].casefold() == "true" and row["expected_decision"] == "answer":
            answer = agent._render(question, evidence, product)
            required_groups = groups(row["must_include"])
            forbidden = [x.strip() for x in (row["must_not_include"] or "").split(";") if x.strip()]
            req_ok = all(contains_alternatives(answer, alt) for alt in required_groups)
            forbidden_ok_case = not any(normalize(x) and normalize(x) in normalize(answer) for x in forbidden)
            valid_cites = re.findall(r"\[([A-Za-z0-9_.:-]+)\]", answer)
            allowed = set(ranked_ids)
            if product:
                allowed.add("products.csv:" + product["sku"])
            cite_ok = bool(valid_cites) and all(c in allowed for c in valid_cites)
            generated_count += 1
            required_ok += req_ok
            forbidden_ok += forbidden_ok_case
            citations_ok += cite_ok
            ans_correct += req_ok and forbidden_ok_case and cite_ok
            record["answer"] = answer
            record["required_facts_pass"] = req_ok
            record["forbidden_phrases_pass"] = forbidden_ok_case
            record["citation_ids_valid"] = cite_ok
            answer_rows.append(record)
        rows.append(record)

    n_evaluable = sum(1 for r in rows if r["evidence_hit_at_k"] is not None)
    report = {
        "dataset": "supplied practice/questions.csv (development set; not held out)",
        "practice_question_count": len(rows), "top_k": args.top_k,
        "evidence_eligible_questions": n_evaluable,
        "evidence_hit_at_k": retrieval_hits / max(1, n_evaluable),
        "complete_gold_evidence_at_k": retrieval_complete / max(1, n_evaluable),
        "mrr_at_k": mrr_total / max(1, n_evaluable),
        "hard_negative_retrieved_count": hard_negative_cases,
        "hard_negative_eligible_questions": hard_negative_eligible,
        "hard_negative_retrieval_rate": hard_negative_cases / max(1, hard_negative_eligible),
        "generated_answer_count": generated_count,
        "required_facts_pass_rate": required_ok / max(1, generated_count) if args.generate else None,
        "forbidden_phrase_pass_rate": forbidden_ok / max(1, generated_count) if args.generate else None,
        "valid_citation_pass_rate": citations_ok / max(1, generated_count) if args.generate else None,
        "strict_heuristic_answer_pass_rate": ans_correct / max(1, generated_count) if args.generate else None,
        "limitations": ["Phrase checks are heuristic, not semantic review.",
                        "Questions are supplied development examples; no hidden set was used.",
                        "Conversation flows, tool-call correctness, and voice quality need separate review."]}
    out_stem = ROOT / args.out
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    out_stem.with_suffix(".json").write_text(json.dumps({"summary": report, "rows": rows},
                                                            ensure_ascii=False, indent=2), encoding="utf-8")
    if args.generate:
        csv_path = out_stem.with_name(out_stem.name + "_answers.csv")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            cols = ["id", "language", "question", "answerable", "expected_decision", "gold_evidence",
                    "retrieved_sections", "answer", "required_facts_pass", "forbidden_phrases_pass", "citation_ids_valid"]
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for item in rows:
                if item["answer"]:
                    writer.writerow({**item, "gold_evidence": ";".join(item["gold_evidence"]),
                                     "retrieved_sections": ";".join(item["retrieved_sections"])})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
