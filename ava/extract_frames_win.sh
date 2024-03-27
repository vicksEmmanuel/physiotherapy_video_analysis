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

  "C:\Users\Q2094871\Downloads\ffmpeg-master-latest-win64-gpl\ffmpeg\bin\ffmpeg.exe" -i "${video}" -r 30 -q:v 1 "${out_name}"
done