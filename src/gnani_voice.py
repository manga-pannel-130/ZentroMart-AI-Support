"""Optional Gnani speech integration for the capstone demo.

Requires GNANI_API_KEY and the `gnani-vachana` package. This is written to the
public SDK interface; a real end-to-end voice run still needs an authorized key.
"""
from __future__ import annotations

import asyncio
import io
import os
import re
from typing import Any
from datetime import date

STT_LANGUAGES = {
    "English (including mixed English/Hindi)": "en-IN", "Hindi": "hi-IN", "Tamil": "ta-IN",
    "Telugu": "te-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN", "Marathi": "mr-IN",
    "Bengali": "bn-IN", "Gujarati": "gu-IN", "Punjabi": "pa-IN",
}


def _wav_to_pcm16_mono(audio_bytes: bytes, target_rate: int = 16000) -> bytes:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    if samples.size == 0 or sample_rate <= 0:
        raise ValueError("The recording is empty or not a readable audio file.")
    mono = samples.mean(axis=1)
    if sample_rate != target_rate:
        from math import gcd
        factor = gcd(int(sample_rate), int(target_rate))
        mono = resample_poly(mono, target_rate // factor, sample_rate // factor)
    mono = np.clip(mono, -1.0, 1.0)
    return (mono * 32767).astype("<i2").tobytes()


def transcribe_recording(audio_bytes: bytes, language: str = "en-IN",
                         api_key: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Transcribe a user recording through Gnani STT."""
    api_key = api_key or os.getenv("GNANI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GNANI_API_KEY in the environment or app secrets to enable voice.")
    from gnani.stt import GnaniSTTStreamClient
    pcm = _wav_to_pcm16_mono(audio_bytes)

    # Browser recordings often end immediately after the last word. Send
    # explicit silence so server-side VAD can close the spoken segment.
    pcm_with_trailing_silence = pcm + bytes(int(16000 * 2 * 1.25))

    async def run() -> list[Any]:
        async with GnaniSTTStreamClient(api_key=api_key, language_code=language,
                                        sample_rate=16000) as stream:
            return await stream.stream_audio(io.BytesIO(pcm_with_trailing_silence), realtime_pace=True)

    events = asyncio.run(run())
    transcripts = [e for e in events if hasattr(e, "text") and getattr(e, "text", "").strip()]
    text = " ".join(e.text.strip() for e in transcripts)
    details = [{"segment_index": getattr(e, "segment_index", None),
                "latency_ms": getattr(e, "latency", None),
                "audio_duration_ms": getattr(e, "audio_duration_ms", None)} for e in transcripts]
    if not text:
        # REST transcription avoids realtime VAD/end-of-speech timing. Use it
        # once as a fallback if the streaming API emitted no transcript.
        try:
            from gnani.stt import GnaniSTTClient
            response = GnaniSTTClient(api_key=api_key).transcribe_bytes(
                audio_bytes, filename="recording.wav", language_code=language
            )
            text = str(response.get("transcript", "")).strip()
            if text:
                details = [{"source": "rest_fallback",
                            "latency_ms": response.get("end_to_end_latency")}]
        except Exception as exc:
            fallback_error = type(exc).__name__
        else:
            fallback_error = "empty transcript"
        if not text:
            raise RuntimeError(
                "Gnani returned no speech transcript from realtime or fallback "
                f"transcription ({fallback_error}). Check that the clip contains "
                "clear speech, select the language spoken, and record at least 2 seconds."
            )
    return text, details


def _under_thousand(number: int) -> str:
    ones = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
            "eighteen", "nineteen")
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
    parts = []
    if number >= 100:
        parts.extend([ones[number // 100], "hundred"])
        number %= 100
    if number >= 20:
        parts.append(tens[number // 10])
        if number % 10:
            parts.append(ones[number % 10])
    elif number:
        parts.append(ones[number])
    return " ".join(parts)


def _indian_number_words(number: int) -> str:
    if number == 0:
        return "zero"
    if number < 0:
        return "minus " + _indian_number_words(-number)
    parts = []
    for size, label in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand"), (100, "hundred")):
        count, number = divmod(number, size)
        if count:
            parts.append(_under_thousand(count) + " " + label)
    if number:
        parts.append(_under_thousand(number))
    return " ".join(parts)


def format_for_voice(text: str) -> str:
    """Apply the course's spoken reply conventions before TTS."""
    text = re.sub(r"\[(?:[A-Z0-9_.:-]+|products\.csv:[A-Z]{3}-\d{4})\]", "", text)
    text = re.sub(r"(?m)^\s*[-•]\s*", "", text)
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"(?<!\d)(\d{4})[ -]?(\d{4})(?!\d)", r"\1 \2", text)
    def date_words(match: re.Match[str]) -> str:
        try:
            d = date.fromisoformat(match.group(0))
            return f"{d.day} {d.strftime('%B')} {d.year}"
        except ValueError:
            return match.group(0)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", date_words, text)
    def rupees(match: re.Match[str]) -> str:
        digits = int(match.group(1).replace(",", ""))
        return _indian_number_words(digits) + " rupees"
    text = re.sub(r"₹\s*([\d,]+)", rupees, text)
    return re.sub(r"\s+", " ", text).strip()


def synthesize_reply(text: str, language: str = "auto", voice: str | None = None) -> bytes:
    """Request streamed Gnani TTS and collect its WAV response for browser playback."""
    api_key = os.getenv("GNANI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GNANI_API_KEY in the environment or app secrets to enable voice.")
    from gnani.tts import GnaniTTSRealtimeClient
    chosen_voice = voice or os.getenv("GNANI_TTS_VOICE", "Kaveri")

    async def run() -> bytes:
        async with GnaniTTSRealtimeClient(api_key=api_key) as client:
            return await client.synthesize_and_collect(format_for_voice(text), voice=chosen_voice,
                                                       model=os.getenv("GNANI_TTS_MODEL", "timbre-v2.5"),
                                                       language=language)
    return asyncio.run(run())


