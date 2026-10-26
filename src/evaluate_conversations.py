"""Replay the visible conversation practice cases against the agent.

Uses the supplied synthetic lookup tool and a dry-run ticket stub. It never
writes practice tickets, and output omits messages and verification values.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from zentro_agent import (ROOT, DATA_ROOT, HybridRetriever, LocalGenerator, ZentroSupportAgent,
                          is_proof, load_sections)


def alternatives(cell: str) -> list[list[str]]:
    groups = []
    for group in (cell or "").split(";"):
        options = [x.strip() for x in group.split("/") if x.strip()]
        if options:
            groups.append(options)
    return groups


def normalize(text: str) -> str:
    out = []
    for char in text.casefold():
        try:
            out.append(str(unicodedata.digit(char)))
        except (TypeError, ValueError):
            out.append(char)
    return " ".join(re.sub(r"[^\w₹]+", " ", "".join(out), flags=re.UNICODE).split())


def fact_check(text: str, cell: str) -> bool:
    return all(any(normalize(option) in normalize(text) for option in group) for group in alternatives(cell))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/conversation_evaluation.json")
    parser.add_argument("--max-conversations", type=int, default=0,
                        help="optional small development run; 0 means all conversations")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    retriever = HybridRetriever(load_sections(), embedder)
    generator = LocalGenerator()
    agent = ZentroSupportAgent(retriever, generator, mock_tools=False)
    from tools.tools import lookup_order
    agent.lookup_order = lookup_order
    agent.create_ticket = lambda **kwargs: {"created": True, "ticket_id": "TKT-DRY-RUN"}
    agent.mock_tools = True

    raw = list(csv.DictReader((DATA_ROOT / "practice" / "conversations.csv").open(encoding="utf-8-sig", newline="")))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row["conversation_id"]].append(row)
    selected = sorted(grouped)[:args.max_conversations] if args.max_conversations else sorted(grouped)
    output_rows = []
    for conversation_id in selected:
        state = {}
        history: list[dict[str, str]] = []
        for row in sorted(grouped[conversation_id], key=lambda item: int(item["turn"])):
            message = row["customer_message"]
            history_input = ("[verification value provided]"
                             if state.get("pending_order_id") and is_proof(message)
                             else message)
            result = agent.respond(message, state, conversation_history=history)
            history.extend([{"role": "user", "content": history_input},
                            {"role": "assistant", "content": result["text"]}])
            del history[:-8]
            state = result.get("state", state)
            expected_tools = set(re.findall(r"\b(lookup_order|create_ticket)\s*\(", row["expected_tool_calls"] or ""))
            actual_tools = set(re.match(r"([a-z_]+)", tool).group(1)
                               for tool in result.get("tool_calls", []) if re.match(r"([a-z_]+)", tool))
            gold = {x.strip() for x in row["gold_evidence"].split(";") if x.strip()}
            found = set(result.get("evidence", []))
            output_rows.append({
                "conversation_id": conversation_id,
                "turn": int(row["turn"]),
                "scenario": row["scenario"],
                "expected_decision": row["expected_decision"],
                "actual_decision": result.get("decision"),
                "decision_pass": result.get("decision") == row["expected_decision"],
                "expected_tool_names": sorted(expected_tools),
                "actual_tool_names": sorted(actual_tools),
                "tool_names_pass": expected_tools == actual_tools,
                "required_facts_pass": fact_check(result["text"], row["must_include"]),
                "forbidden_phrases_pass": not any(normalize(x.strip()) in normalize(result["text"])
                    for x in (row["must_not_include"] or "").split(";") if normalize(x.strip())),
                "gold_evidence": sorted(gold),
                "evidence_recall_pass": (not gold) or bool(gold & found),
            })

    def rate(key: str) -> float:
        return sum(bool(row[key]) for row in output_rows) / max(1, len(output_rows))

    report = {
        "dataset": "supplied practice/conversations.csv (development set; not held out)",
        "conversation_count": len(selected), "turn_count": len(output_rows),
        "decision_accuracy": rate("decision_pass"),
        "tool_name_match_rate": rate("tool_names_pass"),
        "required_fact_pass_rate": rate("required_facts_pass"),
        "forbidden_phrase_pass_rate": rate("forbidden_phrases_pass"),
        "evidence_recall_rate": rate("evidence_recall_pass"),
        "limitations": ["Uses a local generative model; output varies by checkpoint and generation stack.",
                        "Phrase matching is heuristic and cannot grade semantic correctness or language fluency.",
                        "Tool calls are compared by tool name; arguments and policy appropriateness require review.",
                        "Order records and conversations are synthetic practice data, not hidden evaluation."]}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": report, "turns": output_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
