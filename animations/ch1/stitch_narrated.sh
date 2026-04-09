#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
# AIMS Chapter 1 — Build narrated video
#
# Combines per-section narration audio with animation clips.
# When narration is longer than animation, the last frame freezes.
# When there's no animation (opening/closing), extends the title card.
#
# Usage:
#   bash animations/ch1/stitch_narrated.sh
#   bash animations/ch1/stitch_narrated.sh --music animations/music/chopin_nocturne_op9_no2.mp3
#
# Output: animations/ch1/chapter1_narrated.mp4
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
MEDIA="$ROOT/media/ch1/videos"
NAR="$ROOT/animations/ch1/narration"
TITLES="$MEDIA/section_titles/1080p60"
OUT="$ROOT/animations/ch1/chapter1_narrated.mp4"
TMPDIR="$(mktemp -d)"

get_duration() {
    ffprobe -v quiet -show_entries format=duration \
        -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null
}

# ── Section definitions ──────────────────────────────────────────
# Format: "section_id  title_card_video  animation_video  narration_audio"
# Use "NONE" if no animation/title for that section.
SECTIONS=(
    "opening        $TITLES/ChapterOpening.mp4    $MEDIA/opening_hook/1080p60/OpeningHook.mp4                   $NAR/part1_opening.mp3"
    "part1          $TITLES/Part1Title.mp4         $MEDIA/response_matrix/1080p60/ResponseMatrixSort.mp4         $NAR/part1_response_matrix.mp3"
    "part2          $TITLES/Part2Title.mp4         $MEDIA/icc_models/1080p60/ICCModels.mp4                       $NAR/part2_icc_models.mp3"
    "part3a         $TITLES/Part3Title.mp4         $MEDIA/sufficiency/1080p60/Sufficiency.mp4                    $NAR/part3_sufficiency.mp3"
    "part3b         NONE                           $MEDIA/specific_objectivity/1080p60/SpecificObjectivity.mp4   $NAR/part3_specific_objectivity.mp3"
    "part4a         $TITLES/Part4Title.mp4         $MEDIA/elo_dynamics/1080p60/EloDynamics.mp4                   $NAR/part4_elo.mp3"
    "part4b         NONE                           $MEDIA/latent_vs_network/1080p60/LatentVsNetwork.mp4          $NAR/part4_latent_vs_network.mp3"
    "part5          $TITLES/Part5Title.mp4         $MEDIA/factor_model/1080p60/FactorModel.mp4                   $NAR/part5_factor_model.mp3"
    "closing        $TITLES/ChapterClosing.mp4     NONE                                                          $NAR/part6_closing.mp3"
)

echo "Building narrated chapter video..."
echo ""

SEGMENT_FILES=()

for section_line in "${SECTIONS[@]}"; do
    read -r sec_id title_vid anim_vid nar_audio <<< "$section_line"
    echo "── Section: $sec_id"

    nar_dur=$(get_duration "$nar_audio")
    echo "   Narration: ${nar_dur}s"

    segment_out="$TMPDIR/${sec_id}.mp4"

    if [[ "$anim_vid" == "NONE" && "$title_vid" != "NONE" ]]; then
        # No animation — keep the full title card and pad whichever stream is shorter
        title_dur=$(get_duration "$title_vid")
        final_dur=$(python3 -c "print(max(float($nar_dur), float($title_dur)))")
        video_pad=$(python3 -c "print(max(0.0, float($final_dur) - float($title_dur)))")
        echo "   Title card: ${title_dur}s (target ${final_dur}s)"

        ffmpeg -y -i "$title_vid" -i "$nar_audio" \
            -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${video_pad}[v];[1:a]apad,atrim=0:${final_dur}[a]" \
            -map "[v]" -map "[a]" \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k \
            "$segment_out" 2>/dev/null

    elif [[ "$title_vid" != "NONE" && "$anim_vid" != "NONE" ]]; then
        # Title card + animation — concat video, then pad whichever stream is shorter
        anim_dur=$(get_duration "$anim_vid")
        title_dur=$(get_duration "$title_vid")
        total_vid=$(python3 -c "print($title_dur + $anim_dur)")
        echo "   Title: ${title_dur}s + Animation: ${anim_dur}s = ${total_vid}s"

        # First concat title + animation
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
            -c:a aac -b:a 192k \
            "$segment_out" 2>/dev/null

    elif [[ "$title_vid" == "NONE" && "$anim_vid" != "NONE" ]]; then
        # Animation only (no title card) — preserve the full animation and narration
        anim_dur=$(get_duration "$anim_vid")
        final_dur=$(python3 -c "print(max(float($nar_dur), float($anim_dur)))")
        video_pad=$(python3 -c "print(max(0.0, float($final_dur) - float($anim_dur)))")
        echo "   Animation: ${anim_dur}s (target ${final_dur}s)"

        ffmpeg -y -i "$anim_vid" -i "$nar_audio" \
            -filter_complex "[0:v]tpad=stop_mode=clone:stop_duration=${video_pad}[v];[1:a]apad,atrim=0:${final_dur}[a]" \
            -map "[v]" -map "[a]" \
            -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 60 \
            -c:a aac -b:a 192k \
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
    ffmpeg -y -f concat -safe 0 -i "$FINAL_CONCAT" \
        -c copy "$SILENT_OUT" 2>/dev/null

    # Mix background music at low volume under narration
    vid_dur=$(get_duration "$SILENT_OUT")
    echo "Adding background music (volume=${MUSIC_VOL})..."
    ffmpeg -y -i "$SILENT_OUT" -i "$MUSIC_FILE" \
        -filter_complex \
        "[0:a]volume=1.0[voice];[1:a]volume=${MUSIC_VOL},afade=t=in:d=3,afade=t=out:st=$(python3 -c "print(max(0, float($vid_dur) - 4))"):d=4[bg];[bg]apad[bgpad];[bgpad]atrim=0:${vid_dur}[music];[voice][music]amix=inputs=2:duration=first[aout]" \
        -map 0:v -map "[aout]" \
        -c:v copy -c:a aac -b:a 192k \
        "$OUT" 2>/dev/null
else
    ffmpeg -y -f concat -safe 0 -i "$FINAL_CONCAT" \
        -c copy "$OUT" 2>/dev/null
fi

total_dur=$(get_duration "$OUT")
echo ""
echo "════════════════════════════════════════════"
echo "Done: $OUT"
echo "Total duration: ${total_dur}s ($(python3 -c "m,s=divmod(int($total_dur),60); print(f'{m}:{s:02d}')"))"
echo "════════════════════════════════════════════"

# Segment breakdown
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
