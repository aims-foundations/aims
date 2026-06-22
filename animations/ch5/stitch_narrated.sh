#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# AIMS Chapter 5 — Build narrated video
#
# Combines per-section narration audio with animation clips.
# When narration is longer than animation, the last frame freezes.
# When there's no animation (closing), extends the title card.
#
# Usage:
#   bash animations/ch5/stitch_narrated.sh
#   bash animations/ch5/stitch_narrated.sh --music animations/music/<track>.mp3
#
# Output: animations/ch5/chapter5_narrated.mp4
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

for cmd in ffmpeg ffprobe python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Error: $cmd is required but not installed." >&2
        exit 1
    fi
done

# ── parse arguments ──────────────────────────────────────────────
MUSIC_FILE=""
MUSIC_VOL="0.06"  # lower default for narrated video

while [[ $# -gt 0 ]]; do
    case "$1" in
        --music)        MUSIC_FILE="$2"; shift 2 ;;
        --music-volume) MUSIC_VOL="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MEDIA="$ROOT/media/ch5/videos"
NAR="$ROOT/animations/ch5/narration"
TITLES="$MEDIA/section_titles/1080p60"
OUT="$ROOT/animations/ch5/chapter5_narrated.mp4"
TMPDIR="$(mktemp -d)"

get_duration() {
    ffprobe -v quiet -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

# ── Section definitions ──────────────────────────────────────────
# Format: "section_id  title_card_video  animation_video  narration_audio"
# Part 1 prepends the ChapterOpening card onto its title card via a pre-concat
# (handled below by listing two title clips). For simplicity each row carries a
# single title card; ChapterOpening is stitched as its own leading segment.
SECTIONS=(
    "opening    $TITLES/ChapterOpening.mp4   NONE                                                                        NONE"
    "part1      $TITLES/Part1Title.mp4       $MEDIA/leaderboard_flicker/1080p60/LeaderboardFlicker.mp4                    $NAR/part1_flicker.mp3"
    "part2      $TITLES/Part2Title.mp4       $MEDIA/variance_decomposition/1080p60/VarianceDecomposition.mp4             $NAR/part2_decomp.mp3"
    "part3      $TITLES/Part3Title.mp4       $MEDIA/three_estimators/1080p60/ThreeEstimators.mp4                         $NAR/part3_estimators.mp3"
    "part4      $TITLES/Part4Title.mp4       $MEDIA/conditional_reliability/1080p60/ConditionalReliability.mp4           $NAR/part4_conditional.mp3"
    "part5      $TITLES/Part5Title.mp4       $MEDIA/g_d_study/1080p60/GandDStudy.mp4                                     $NAR/part5_gdstudy.mp3"
    "part6      $TITLES/Part6Title.mp4       $MEDIA/judge_kappa/1080p60/JudgeKappa.mp4                                   $NAR/part6_kappa.mp3"
    "part7      $TITLES/Part7Title.mp4       $MEDIA/spearman_brown/1080p60/SpearmanBrown.mp4                             $NAR/part7_spearman.mp3"
    "closing    $TITLES/ChapterClosing.mp4   NONE                                                                        $NAR/part8_closing.mp3"
)

echo "Building narrated chapter video..."
echo ""

SEGMENT_FILES=()

for section_line in "${SECTIONS[@]}"; do
    read -r sec_id title_vid anim_vid nar_audio <<< "$section_line"
    echo "── Section: $sec_id"

    segment_out="$TMPDIR/${sec_id}.mp4"

    # Opening card has no narration — just re-encode it as its own segment.
    if [[ "$nar_audio" == "NONE" ]]; then
        ffmpeg -y -i "$title_vid" -f lavfi -i anullsrc=r=48000:cl=stereo \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k -ar 48000 -ac 2 -shortest \
            "$segment_out" 2>/dev/null
        seg_dur=$(get_duration "$segment_out")
        echo "   Output: ${seg_dur}s (no narration)"
        echo ""
        SEGMENT_FILES+=("$segment_out")
        continue
    fi

    nar_dur=$(get_duration "$nar_audio")
    echo "   Narration: ${nar_dur}s"

    if [[ "$anim_vid" == "NONE" && "$title_vid" != "NONE" ]]; then
        # No animation — keep the full title card, pad shorter stream
        title_dur=$(get_duration "$title_vid")
        final_dur=$(python3 -c "print(max(float($nar_dur), float($title_dur)))")
        video_pad=$(python3 -c "print(max(0.0, float($final_dur) - float($title_dur)))")
        echo "   Title card: ${title_dur}s (target ${final_dur}s)"

        ffmpeg -y -i "$title_vid" -i "$nar_audio" \
            -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${video_pad}[v];[1:a]apad,atrim=0:${final_dur}[a]" \
            -map "[v]" -map "[a]" \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k -ar 48000 -ac 2 \
            "$segment_out" 2>/dev/null

    elif [[ "$title_vid" != "NONE" && "$anim_vid" != "NONE" ]]; then
        # Title card + animation — concat video, pad shorter stream
        anim_dur=$(get_duration "$anim_vid")
        title_dur=$(get_duration "$title_vid")
        echo "   Title: ${title_dur}s + Animation: ${anim_dur}s"

        concat_list="$TMPDIR/${sec_id}_concat.txt"
        echo "file '$title_vid'" > "$concat_list"
        echo "file '$anim_vid'" >> "$concat_list"
        concat_vid="$TMPDIR/${sec_id}_concat.mp4"

        ffmpeg -y -f concat -safe 0 -i "$concat_list" \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            "$concat_vid" 2>/dev/null

        concat_dur=$(get_duration "$concat_vid")
        final_dur=$(python3 -c "print(max(float($nar_dur), float($concat_dur)))")
        video_pad=$(python3 -c "print(max(0.0, float($final_dur) - float($concat_dur)))")
        echo "   Target segment duration: ${final_dur}s"

        ffmpeg -y -i "$concat_vid" -i "$nar_audio" \
            -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${video_pad}[v];[1:a]apad,atrim=0:${final_dur}[a]" \
            -map "[v]" -map "[a]" \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k -ar 48000 -ac 2 \
            "$segment_out" 2>/dev/null
    fi

    seg_dur=$(get_duration "$segment_out")
    echo "   Output: ${seg_dur}s"
    echo ""
    SEGMENT_FILES+=("$segment_out")
done

# ── Concatenate all segments ─────────────────────────────────────
echo "Concatenating ${#SEGMENT_FILES[@]} segments..."
FINAL_CONCAT="$TMPDIR/final_concat.txt"
for seg in "${SEGMENT_FILES[@]}"; do
    echo "file '$seg'" >> "$FINAL_CONCAT"
done

if [[ -n "$MUSIC_FILE" ]]; then
    SILENT_OUT="$TMPDIR/narrated_silent.mp4"
    ffmpeg -y -f concat -safe 0 -i "$FINAL_CONCAT" -c copy "$SILENT_OUT" 2>/dev/null

    vid_dur=$(get_duration "$SILENT_OUT")
    fade_st=$(python3 -c "print(max(0, float($vid_dur) - 4))")
    echo "Adding background music (volume=${MUSIC_VOL}, looped to ${vid_dur}s)..."
    # -stream_loop -1 loops the music so it covers the full (longer) video;
    # fades are applied on the continuous stream, then trimmed to vid_dur.
    # amix normalize=0 keeps the narration at full level under the quiet music.
    ffmpeg -y -i "$SILENT_OUT" -stream_loop -1 -i "$MUSIC_FILE" \
        -filter_complex \
        "[0:a]volume=1.0[voice];[1:a]volume=${MUSIC_VOL},afade=t=in:d=3,afade=t=out:st=${fade_st}:d=4,atrim=0:${vid_dur}[music];[voice][music]amix=inputs=2:duration=first:normalize=0[aout]" \
        -map 0:v -map "[aout]" \
        -c:v copy -c:a aac -b:a 192k \
        "$OUT" 2>/dev/null
else
    ffmpeg -y -f concat -safe 0 -i "$FINAL_CONCAT" -c copy "$OUT" 2>/dev/null
fi

total_dur=$(get_duration "$OUT")
echo ""
echo "════════════════════════════════════════════"
echo "Done: $OUT"
echo "Total duration: ${total_dur}s ($(python3 -c "m,s=divmod(int($total_dur),60); print(f'{m}:{s:02d}')"))"
echo "════════════════════════════════════════════"

echo ""
echo "Segment breakdown:"
echo "──────────────────────────────────────────"
for seg in "${SEGMENT_FILES[@]}"; do
    d=$(get_duration "$seg")
    printf "  %-30s %6.1fs\n" "$(basename "$seg" .mp4)" "$d"
done
echo "──────────────────────────────────────────"
printf "  %-30s %6.1fs\n" "Total" "$total_dur"

rm -rf "$TMPDIR"
