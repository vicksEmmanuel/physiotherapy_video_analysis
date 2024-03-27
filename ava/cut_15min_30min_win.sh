IN_DATA_DIR="videos"
OUT_DATA_DIR="videos_15min"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR}
fi

for video in $(ls -A1 -U ${IN_DATA_DIR}/*)
do
  out_name="${OUT_DATA_DIR}/${video##*/}"
  if [ ! -f "${out_name}" ]; then
    "C:\Users\Q2094871\Downloads\ffmpeg-master-latest-win64-gpl\ffmpeg\bin\ffmpeg.exe" -ss 900 -t 901 -i "${video}" "${out_name}"
  fi
done