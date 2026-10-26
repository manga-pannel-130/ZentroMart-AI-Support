"""Batch runner required by VoiceOps: transcribe, answer, synthesize and save evidence."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.as_posix())]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a folder of customer audio using Gnani STT/TTS and the ZentroMart agent.")
    parser.add_argument("--audio-dir", required=True, type=Path,
                        help="Folder containing WAV, FLAC, OGG, or AIFF customer turns")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Where to save reply audio and transcript/evidence manifests")
    parser.add_argument("--data-dir", default=None, type=Path,
                        help="Extracted dataset root; defaults to ZENTROMART_DATA_DIR or automatic discovery")
    parser.add_argument("--stt-language", default="en-IN",
                        choices=["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN",
                                 "mr-IN", "bn-IN", "gu-IN", "pa-IN"],
                        help="Gnani STT language; en-IN supports mixed English/Hindi")
    parser.add_argument("--tts-language", default="auto",
                        choices=["auto", "en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN",
                                 "mr-IN", "bn-IN", "gu-IN", "pa-IN", "hi-en"])
    parser.add_argument("--model-id", default=None, help="Optional Hugging Face instruct model override")
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not audio_dir.is_dir():
        parser.error(f"Audio folder does not exist: {audio_dir}")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    from data_paths import resolve_data_root
    try:
        data_dir = resolve_data_root(args.data_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    # Check again after loading the local, untracked .env file.
    if not os.getenv("GNANI_API_KEY"):
        parser.error("GNANI_API_KEY is required. Add it to a local .env file or set it in the environment.")
    # Set the dataset path before importing the agent and its offline tools.
    os.environ["ZENTROMART_DATA_DIR"] = str(data_dir)

    supported = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
    audio_files = sorted((p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported),
                         key=natural_key)
    if not audio_files:
        parser.error(f"No supported audio files found under {audio_dir}.")

    from sentence_transformers import SentenceTransformer
    from zentro_agent import HybridRetriever, LocalGenerator, ZentroSupportAgent, is_proof, load_sections
    from gnani_voice import transcribe_recording, synthesize_reply

    embedder = SentenceTransformer(os.getenv("EMBED_MODEL_ID", "intfloat/multilingual-e5-small"), device="cpu")
    retriever = HybridRetriever(load_sections(), embedder, top_k=6)
    agent = ZentroSupportAgent(retriever, LocalGenerator(args.model_id or os.getenv("LLM_MODEL_ID")))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    states: dict[str, dict] = {}
    histories: dict[str, list[dict[str, str]]] = {}
    for index, audio_path in enumerate(audio_files, start=1):
        relative_audio = audio_path.relative_to(audio_dir).as_posix()
        conversation_id = re.sub(r"_turn_\d+$", "", audio_path.stem, flags=re.I)
        state = states.setdefault(conversation_id, {})
        history = histories.setdefault(conversation_id, [])
        audio_name = ""
        status = "ok"
        try:
            try:
                transcript, _stt_details = transcribe_recording(audio_path.read_bytes(), args.stt_language)
            except Exception as exc:
                transcript = ""
                status = f"stt_error: {type(exc).__name__}: {exc}"
            if transcript.strip():
                try:
                    history_input = ("[verification value provided]"
                                     if state.get("pending_order_id") and is_proof(transcript)
                                     else transcript)
                    result = agent.respond(transcript, state, conversation_history=history)
                    states[conversation_id] = result.get("state", state)
                    history.extend([{"role": "user", "content": history_input},
                                    {"role": "assistant", "content": result["text"]}])
                    del history[:-8]
                    reply = result["text"]
                    evidence = result.get("evidence", [])
                    retrieved_chunks = result.get("retrieved_chunks", [])
                    decision = result.get("decision", "answer")
                    tool_calls = result.get("tool_calls", [])
                    status = "ok"
                except Exception as exc:
                    reply = "Sorry, I couldn't prepare a reliable answer. Could you please try again or ask for a colleague?"
                    evidence, retrieved_chunks, decision, tool_calls = [], [], "error", []
                    status = f"agent_error: {type(exc).__name__}: {exc}"
            else:
                transcript = ""
                reply = "Sorry, I couldn't catch that. Could you please say it again?"
                evidence, retrieved_chunks, decision, tool_calls = [], [], "clarify", []
                if not status.startswith("stt_error:"):
                    status = "empty_transcript"
            try:
                reply_audio = synthesize_reply(reply, language=args.tts_language)
                audio_name = f"{index:04d}_{audio_path.stem}.wav"
                (output_dir / "audio" / audio_name).write_bytes(reply_audio)
            except Exception as exc:
                tts_error = f"tts_error: {type(exc).__name__}: {exc}"
                status = f"{status}; {tts_error}" if status != "ok" else tts_error
        except Exception as exc:
            # Preserve a failure row instead of silently dropping an input.
            transcript, reply, evidence, retrieved_chunks, decision, tool_calls = "", "", [], [], "error", []
            status = f"error: {type(exc).__name__}: {exc}"
        record = {"index": index, "conversation_id": conversation_id, "input_audio": relative_audio,
                  "transcript": transcript, "reply_text": reply, "reply_audio": f"audio/{audio_name}" if audio_name else "",
                  "retrieved_sections": evidence, "retrieved_chunks": retrieved_chunks,
                  "decision": decision, "tool_calls": tool_calls, "status": status}
        records.append(record)
        print(f"[{index}/{len(audio_files)}] {relative_audio}: {status}")

    (output_dir / "turns.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "turns.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["index", "conversation_id", "input_audio", "transcript", "reply_text", "reply_audio",
                   "retrieved_sections", "retrieved_chunks", "decision", "tool_calls", "status"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({**record,
                             "retrieved_sections": ";".join(record["retrieved_sections"]),
                             "retrieved_chunks": json.dumps(record["retrieved_chunks"], ensure_ascii=False),
                             "tool_calls": ";".join(record["tool_calls"])})
    print(f"Saved {len(records)} turn records to {output_dir}")


if __name__ == "__main__":
    main()
