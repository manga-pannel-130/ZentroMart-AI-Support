"""Streamlit UI for the ZentroMart capstone assistant."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(APP_ROOT / ".env", override=False)
except ImportError:
    pass

st.set_page_config(page_title="ZentroMart Support Assistant", page_icon="🛍️", layout="centered")

from data_paths import resolve_data_root
try:
    os.environ["ZENTROMART_DATA_DIR"] = str(resolve_data_root())
except FileNotFoundError as exc:
    st.error("The capstone dataset could not be found or is incomplete.")
    st.info("Set ZENTROMART_DATA_DIR to the extracted dataset folder containing kb/, data/, practice/, and tools/.")
    st.code(str(exc))
    st.stop()

from zentro_agent import LocalGenerator, HybridRetriever, ZentroSupportAgent, is_proof, load_sections


def secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


@st.cache_resource(show_spinner="Loading multilingual product and policy search…")
def create_agent() -> ZentroSupportAgent:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(secret("EMBED_MODEL_ID", "intfloat/multilingual-e5-small"), device="cpu")
    retriever = HybridRetriever(load_sections(), embedder, top_k=6)
    generator = LocalGenerator(secret("LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    return ZentroSupportAgent(retriever, generator)


def respond(message: str) -> None:
    assistant = st.session_state.agent
    verification_reply = bool(st.session_state.agent_state.get("pending_order_id") and is_proof(message))
    conversation_history = [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[-8:]
        if item.get("role") in {"user", "assistant"}
    ]
    st.session_state.pop("voice_error", None)
    result = assistant.respond(message, st.session_state.agent_state,
                               conversation_history=conversation_history)
    st.session_state.agent_state = result.get("state", st.session_state.agent_state)
    displayed_message = "[verification value provided]" if verification_reply else message
    st.session_state.messages.append({"role": "user", "content": displayed_message})
    st.session_state.messages.append({"role": "assistant", "content": result["text"],
                                      "evidence": result.get("evidence", []),
                                      "decision": result.get("decision"),
                                      "tool_calls": result.get("tool_calls", [])})
    if st.session_state.speak_replies and secret("GNANI_API_KEY"):
        try:
            from gnani_voice import synthesize_reply
            audio = synthesize_reply(result["text"], language="auto")
            st.session_state.messages[-1]["audio"] = audio
            st.session_state.messages[-1]["audio_played"] = False
        except Exception as exc:
            st.session_state.voice_error = str(exc)


st.title("ZentroMart Support Assistant")
st.caption("Capstone demo · grounded answers · cited sources · verified order lookup")

with st.sidebar:
    st.subheader("Demo settings")
    from gnani_voice import STT_LANGUAGES
    voice_language_label = st.selectbox("Voice message language", list(STT_LANGUAGES), index=0)
    speak_choice = st.checkbox("Speak replies with Gnani TTS", value=True, key="speak_replies_v2",
                               help="Enabled by default. Requires GNANI_API_KEY and the Gnani SDK.")
    st.session_state.speak_replies = speak_choice
    st.caption("The included order and customer records are fictional. The mock identity check is for this course only.")
    if secret("GNANI_API_KEY"):
        st.success("Gnani voice key found")
    else:
        st.info("Text mode is ready without a Gnani key. Voice options need GNANI_API_KEY.")
    if st.button("Start a new conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_state = {}
        st.session_state.voice_transcript = ""
        st.session_state.last_audio_hash = None
        st.session_state.processed_audio_hash = None
        st.session_state.pop("voice_recording", None)
        st.rerun()

if "agent" not in st.session_state:
    try:
        st.session_state.agent = create_agent()
    except Exception as exc:
        st.error("The assistant could not load its local models or knowledge base.")
        st.exception(exc)
        st.stop()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {}
if "speak_replies" not in st.session_state:
    st.session_state.speak_replies = True
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "processed_audio_hash" not in st.session_state:
    st.session_state.processed_audio_hash = None

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant" and item.get("evidence"):
            decision = item.get("decision")
            if decision == "catalog_list":
                st.caption(f"Source: product catalog ({len(item['evidence'])} listings)")
            elif decision == "catalog_overview":
                st.caption("Source: product catalog")
            else:
                st.caption("Evidence IDs retrieved: " + ", ".join(item["evidence"]))
        if item["role"] == "assistant" and item.get("tool_calls"):
            with st.expander("Tool and safety trace"):
                st.write("Decision:", item.get("decision"))
                st.write("Tool calls:", item["tool_calls"])
        if item["role"] == "assistant" and item.get("audio"):
            autoplay = not item.get("audio_played", False)
            st.audio(item["audio"], format="audio/wav", autoplay=autoplay)
            item["audio_played"] = True
if st.session_state.get("voice_error"):
    st.warning("Gnani TTS was unavailable for this reply: " + st.session_state.pop("voice_error"))

audio_input = st.audio_input("Record a voice message", key="voice_recording")
if audio_input is not None:
    audio_bytes = audio_input.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()
    if digest != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = digest
        st.session_state.voice_transcript = ""
        st.session_state.pop("voice_error", None)

    if st.session_state.processed_audio_hash == digest:
        st.caption("This voice message has been sent. Record a new message to continue.")
    elif not st.session_state.voice_transcript:
        if not secret("GNANI_API_KEY"):
            st.warning("Set GNANI_API_KEY in Streamlit secrets or the environment to transcribe voice.")
        elif st.button("Transcribe recording", type="primary", key=f"transcribe_{digest[:12]}"):
            try:
                from gnani_voice import transcribe_recording
                with st.spinner("Transcribing your recording with Gnani…"):
                    transcript, segments = transcribe_recording(
                        audio_bytes, STT_LANGUAGES[voice_language_label],
                        api_key=secret("GNANI_API_KEY"))
                st.session_state.voice_transcript = transcript
                st.session_state.voice_segment_count = len(segments)
            except Exception as exc:
                st.session_state.voice_error = str(exc)
    if st.session_state.get("voice_error"):
        st.error("Gnani STT could not transcribe this recording: " + st.session_state.pop("voice_error"))

if st.session_state.voice_transcript and st.session_state.processed_audio_hash != st.session_state.last_audio_hash:
    st.subheader("Review the transcript")
    st.caption(f"Transcript received in {st.session_state.get('voice_segment_count', 0)} segment(s). Review it before sending.")
    transcript_key = f"voice_transcript_edit_{st.session_state.last_audio_hash[:12]}"
    st.text_area("Edit transcription if needed", value=st.session_state.voice_transcript,
                 key=transcript_key)
    if st.button("Send voice message", type="primary", key=f"send_voice_{st.session_state.last_audio_hash[:12]}"):
        voice_message = st.session_state[transcript_key].strip()
        if voice_message:
            with st.spinner("Checking the knowledge base and preparing a reply…"):
                respond(voice_message)
            st.session_state.voice_transcript = ""
            st.session_state.processed_audio_hash = st.session_state.last_audio_hash
            st.rerun()

prompt = st.chat_input("Ask about a policy, product, or order")
if prompt:
    with st.spinner("Checking the knowledge base and preparing a reply…"):
        respond(prompt)
    st.rerun()

if not st.session_state.messages:
    st.info("Try a policy question, a product SKU such as ELE-0012, or an order ID. Order details require verification.")
