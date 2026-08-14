#!/usr/bin/env bash
set -euo pipefail

video_dir=''
output_dir='../data/{dataset_name}'

mkdir -p "$output_dir"

for video in "$video_dir"/*.mp4; do
  [[ -e "$video" ]] || continue
  video_name="$(basename "$video" .mp4)"
  out_video_dir="$output_dir/$video_name"
  mkdir -p "$out_video_dir"

  echo "Processing video: $video_name"

  duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video")"
  duration_seconds="$(awk -v d="$duration" 'BEGIN { print int(d) }')"
  segments=$(( (duration_seconds + 29) / 30 ))   # ceil(duration/30)

  echo "  Duration: ${duration_seconds}s, Segments: $segments"

  for ((i=0; i<segments; i++)); do
    start=$((i * 30))
    output="$out_video_dir/$i.mp4"

    [[ -f "$output" ]] && continue

    ffmpeg -nostdin -hide_banner -loglevel error -y \
      -ss "$start" -i "$video" -t 30 \
      -c:v libx264 -pix_fmt yuv420p -preset medium -crf 23 \
      -c:a aac -b:a 128k \
      -movflags +faststart \
      "$output"
  done

  echo "  Completed: $video_name"
  echo
done

echo "All videos processed successfully!"