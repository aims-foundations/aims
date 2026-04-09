"""
Generate TTS narration from script.md using edge-tts.

Produces per-section audio files aligned to animation clip durations.
Text is split into chunks at pause markers, with real silence inserted
between them via ffmpeg.

Usage:
    python animations/ch2/generate_narration.py
    python animations/ch2/generate_narration.py --voice en-US-GuyNeural
    python animations/ch2/generate_narration.py --rate "+10%"

Output: animations/ch2/narration/<section>.mp3
"""

import asyncio
import argparse
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List

SCRIPT_PATH = "animations/ch2/script.md"
NARRATION_DIR = "animations/ch2/narration"

# ── Section definitions ──────────────────────────────────────────
# Each section maps to an animation clip. We extract the narrator
# text and pair it with the target animation duration so the TTS
# rate can be adjusted per-section to roughly match.
#
# animation_dur is the clip duration in seconds (0 = no animation)
# NOTE: Update these after rendering all animations with actual durations!
SECTION_DEFS = [
    # (section_id, script_start, script_end, animation_dur)
    ("part1_opening",
     "### 1.1 Opening Hook",
     "### 1.2 Maximum Likelihood",
     40.4),  # ChapterOpening (6.6s) + OpeningHook (33.8s)

    ("part1_likelihood",
     "### 1.2 Maximum Likelihood",
     "## PART 2",
     53.0),  # Part1Title (4.6s) + LikelihoodLandscape (48.4s)

    ("part2_identifiability",
     "### 2.1 The Identifiability Problem",
     "## PART 3",
     50.9),  # Part2Title (4.6s) + Identifiability (46.3s)

    ("part3_em",
     "### 3.1 The EM Framework",
     "## PART 4",
     62.8),  # Part3Title (4.6s) + EMAlgorithm (58.2s)

    ("part4_bayesian",
     "### 4.1 Prior, Likelihood, Posterior",
     "## PART 5",
     48.6),  # Part4Title (4.6s) + BayesianInference (44.0s)

    ("part5_cat",
     "### 5.1 Fisher Information and CAT",
     "## PART 6",
     52.3),  # Part5Title (4.6s) + CATSimulation (47.7s)

    ("part6_closing",
     "### 6.1 Summary",
     "## Animation-Scene Mapping",
     62.7),  # ChapterClosing (62.7s)
]


def extract_between(content: str, start: str, end: str) -> str:
    s = content.find(start)
    e = content.find(end)
    if s == -1:
        return ""
    if e == -1:
        e = len(content)
    return content[s:e]


def parse_narration(text: str) -> List[Dict[str, Any]]:
    """Parse narration text into a list of chunks and pauses.

    Returns list of:
        {"type": "text", "content": "..."}
        {"type": "pause", "duration_ms": 1000}
    """
    lines = text.split("\n")
    chunks = []
    current_text = []

    def flush_text():
        t = " ".join(current_text).strip()
        if t:
            chunks.append({"type": "text", "content": t})
        current_text.clear()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("###") or stripped == "**NARRATOR:**" or stripped.startswith("---"):
            continue

        if stripped.startswith(">"):
            if "Cue:" in stripped or "ANIMATION:" in stripped:
                flush_text()
                chunks.append({"type": "pause", "duration_ms": 800})
            continue

        if stripped == "[pause]":
            flush_text()
            chunks.append({"type": "pause", "duration_ms": 700})
            continue

        if stripped == "[beat]":
            flush_text()
            chunks.append({"type": "pause", "duration_ms": 400})
            continue

        if stripped.startswith("|") or stripped.startswith("```") or stripped.startswith("- **"):
            continue

        if not stripped:
            flush_text()
            chunks.append({"type": "pause", "duration_ms": 250})
            continue

        # Clean markdown
        cleaned = stripped
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
        cleaned = cleaned.replace("`", "")
        cleaned = cleaned.replace("> ", "")

        # Pronunciation fixes for Chapter 2
        cleaned = cleaned.replace("theta-i", "theta i")
        cleaned = cleaned.replace("theta_i", "theta i")
        cleaned = cleaned.replace("beta-j", "beta j")
        cleaned = cleaned.replace("beta_j", "beta j")
        cleaned = cleaned.replace("1PL", "one P L")
        cleaned = cleaned.replace("2PL", "two P L")
        cleaned = cleaned.replace("3PL", "three P L")
        cleaned = cleaned.replace("GPT-4", "G P T 4")
        cleaned = cleaned.replace("MMLU", "M M L U")
        # Chapter 2 specific
        cleaned = cleaned.replace("MLE", "M L E")
        cleaned = cleaned.replace("MAP", "M A P")
        cleaned = cleaned.replace("MCMC", "M C M C")
        cleaned = cleaned.replace("L-BFGS", "L B F G S")
        cleaned = cleaned.replace("JMLE", "J M L E")
        cleaned = cleaned.replace("CMLE", "C M L E")
        cleaned = cleaned.replace("MMLE", "M M L E")
        # Be careful with "EM" — only replace standalone
        cleaned = re.sub(r'\bEM\b', 'E M', cleaned)
        # CAT as standalone
        cleaned = re.sub(r'\bCAT\b', 'C A T', cleaned)
        cleaned = cleaned.replace("IRT", "I R T")

        if cleaned.strip():
            current_text.append(cleaned)

    flush_text()
    return chunks


def get_audio_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def generate_silence(duration_s: float, output: str):
    """Generate a silent audio file."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"anullsrc=r=24000:cl=mono",
         "-t", str(duration_s), "-c:a", "libmp3lame", "-q:a", "9",
         output],
        capture_output=True,
        check=True,
    )


async def generate_tts(text: str, voice: str, rate: str, output: str):
    try:
        import edge_tts
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "edge_tts is required to generate narration. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(output)


async def build_section(section_id: str, chunks: List[Dict[str, Any]],
                        voice: str, rate: str, tmpdir: str, output: str):
    """Build a single section audio file from text chunks + silence gaps."""
    part_files = []
    idx = 0

    for chunk in chunks:
        if chunk["type"] == "pause":
            dur_s = chunk["duration_ms"] / 1000.0
            silence_path = os.path.join(tmpdir, f"{section_id}_{idx:03d}_silence.mp3")
            generate_silence(dur_s, silence_path)
            part_files.append(silence_path)

        elif chunk["type"] == "text":
            tts_path = os.path.join(tmpdir, f"{section_id}_{idx:03d}_tts.mp3")
            await generate_tts(chunk["content"], voice, rate, tts_path)
            part_files.append(tts_path)

        idx += 1

    if not part_files:
        return

    # Concatenate all chunks
    concat_list = os.path.join(tmpdir, f"{section_id}_concat.txt")
    with open(concat_list, "w") as f:
        for pf in part_files:
            f.write(f"file '{pf}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_list, "-c:a", "libmp3lame", "-q:a", "2",
         output],
        capture_output=True,
        check=True,
    )


def estimate_rate(text_chunks: List[Dict[str, Any]], target_dur: float,
                  base_wpm: float = 155) -> str:
    """Estimate a TTS rate adjustment to fit narration into target duration.

    base_wpm: typical speaking rate for edge-tts at +0%.
    Returns a rate string like "+15%" or "+30%".
    """
    # Count words in text chunks
    words = sum(
        len(c["content"].split())
        for c in text_chunks if c["type"] == "text"
    )
    # Count pause time
    pause_s = sum(
        c["duration_ms"] / 1000.0
        for c in text_chunks if c["type"] == "pause"
    )

    if words == 0 or target_dur <= 0:
        return "+0%"

    # Time available for speech
    speech_time = max(target_dur - pause_s, 10)

    # Required WPM
    required_wpm = (words / speech_time) * 60

    # Rate adjustment as percentage
    rate_pct = ((required_wpm / base_wpm) - 1) * 100

    # Clamp to reasonable range: don't go slower than -10% or faster than +60%
    rate_pct = max(-10, min(60, rate_pct))

    sign = "+" if rate_pct >= 0 else ""
    return f"{sign}{int(rate_pct)}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="en-US-AndrewNeural")
    parser.add_argument("--rate", default=None,
                        help="Fixed rate (overrides auto-fit). e.g. '+10%%'")
    parser.add_argument("--output-dir", default=NARRATION_DIR)
    args = parser.parse_args()

    with open(SCRIPT_PATH) as f:
        content = f.read()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Voice: {args.voice}")
    print(f"Rate: {'auto-fit' if args.rate is None else args.rate}")
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        for section_id, start, end, anim_dur in SECTION_DEFS:
            raw = extract_between(content, start, end)
            chunks = parse_narration(raw)

            words = sum(len(c["content"].split())
                        for c in chunks if c["type"] == "text")

            # Determine rate
            if args.rate is not None:
                rate = args.rate
            else:
                rate = estimate_rate(chunks, anim_dur)

            output = os.path.join(args.output_dir, f"{section_id}.mp3")
            print(f"── {section_id}")
            print(f"   {words} words, animation={anim_dur}s, rate={rate}")

            asyncio.run(build_section(
                section_id, chunks, args.voice, rate, tmpdir, output
            ))

            dur = get_audio_duration(output)
            diff = dur - anim_dur
            fit = "OK" if abs(diff) < 5 else ("LONG" if diff > 0 else "SHORT")
            print(f"   -> {dur:.1f}s (delta={diff:+.1f}s) [{fit}]")
            print()

    print("Done! Files in", args.output_dir)


if __name__ == "__main__":
    main()
