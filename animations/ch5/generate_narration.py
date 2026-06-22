"""
Generate TTS narration from script.md using edge-tts.

Produces per-section audio files aligned to animation clip durations.
Text is split into chunks at pause markers, with real silence inserted
between them via ffmpeg.

Usage:
    python animations/ch5/generate_narration.py
    python animations/ch5/generate_narration.py --voice en-US-GuyNeural
    python animations/ch5/generate_narration.py --rate "+10%"

Output: animations/ch5/narration/<section>.mp3
"""

import asyncio
import argparse
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List

SCRIPT_PATH = "animations/ch5/script.md"
NARRATION_DIR = "animations/ch5/narration"

# ── Section definitions ──────────────────────────────────────────
# Each section maps to an animation clip (title card + content scene).
# animation_dur is the clip duration in seconds (0 = no animation).
# NOTE: Update these after rendering all animations with actual durations!
SECTION_DEFS = [
    # (section_id, script_start, script_end, animation_dur)
    ("part1_flicker",
     "### 1.1 Opening Hook",
     "## PART 2",
     50.0),   # ChapterOpening + Part1Title + LeaderboardFlicker

    ("part2_decomp",
     "### 2.1 Variance Components",
     "## PART 3",
     65.0),   # Part2Title + VarianceDecomposition

    ("part3_estimators",
     "### 3.1 Three Estimators",
     "## PART 4",
     75.0),   # Part3Title + ThreeEstimators

    ("part4_conditional",
     "### 4.1 Conditional Reliability",
     "## PART 5",
     45.0),   # Part4Title + ConditionalReliability

    ("part5_gdstudy",
     "### 5.1 G-studies and D-studies",
     "## PART 6",
     60.0),   # Part5Title + GandDStudy

    ("part6_kappa",
     "### 6.1 Cohen's Kappa and Systematic Bias",
     "## PART 7",
     55.0),   # Part6Title + JudgeKappa

    ("part7_spearman",
     "### 7.1 Spearman-Brown and SEM",
     "## PART 8",
     40.0),   # Part7Title + SpearmanBrown

    ("part8_closing",
     "### 8.1 Summary",
     "## Animation-Scene Mapping",
     60.0),   # ChapterClosing
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
    """Parse narration text into a list of text chunks and pauses."""
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

        # Pronunciation fixes for Chapter 5
        cleaned = cleaned.replace("theta_i", "theta i")
        cleaned = cleaned.replace("beta_j", "beta j")
        cleaned = cleaned.replace("LLM", "L L M")
        cleaned = cleaned.replace("MMLU", "M M L U")
        cleaned = cleaned.replace("GPT-4", "G P T 4")
        # Chapter 5 specific
        cleaned = cleaned.replace("ANOVA", "an-OH-va")
        cleaned = cleaned.replace("SEM", "S E M")
        cleaned = re.sub(r"\bIRT\b", "I R T", cleaned)
        cleaned = re.sub(r"\bG-study\b", "G study", cleaned)
        cleaned = re.sub(r"\bD-study\b", "D study", cleaned)
        cleaned = re.sub(r"\bG- and D-studies\b", "G and D studies", cleaned)

        if cleaned.strip():
            current_text.append(cleaned)

    flush_text()
    return chunks


def get_audio_duration(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def generate_silence(duration_s: float, output: str):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
         "-t", str(duration_s), "-c:a", "libmp3lame", "-q:a", "9", output],
        capture_output=True, check=True,
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

    concat_list = os.path.join(tmpdir, f"{section_id}_concat.txt")
    with open(concat_list, "w") as f:
        for pf in part_files:
            f.write(f"file '{pf}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c:a", "libmp3lame", "-q:a", "2", output],
        capture_output=True, check=True,
    )


def estimate_rate(text_chunks: List[Dict[str, Any]], target_dur: float,
                  base_wpm: float = 155) -> str:
    """Estimate a TTS rate adjustment to fit narration into target duration."""
    words = sum(len(c["content"].split())
                for c in text_chunks if c["type"] == "text")
    pause_s = sum(c["duration_ms"] / 1000.0
                  for c in text_chunks if c["type"] == "pause")

    if words == 0 or target_dur <= 0:
        return "+0%"

    speech_time = max(target_dur - pause_s, 10)
    required_wpm = (words / speech_time) * 60
    rate_pct = ((required_wpm / base_wpm) - 1) * 100
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

            rate = args.rate if args.rate is not None else estimate_rate(chunks, anim_dur)

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
