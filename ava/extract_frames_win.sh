IN_DATA_DIR="videos_15min"
OUT_DATA_DIR="frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  video_name=$(basename "$video" .webm)
  video_name=$(basename "$video_name" .mp4)

  out_video_dir=${OUT_DATA_DIR}/${video_name}/
  mkdir -p "${out_video_dir}"

  out_name="${out_video_dir}/${video_name}_%06d.jpg"

  # winget install --id=Gyan.FFmpeg  -e
  ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
done