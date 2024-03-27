DATA_DIR="videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

# wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

while IFS= read -r line; do
  cleanedLine=$(echo "$line" | tr -d '\r')
  wget "https://s3.amazonaws.com/ava-dataset/trainval/$cleanedLine" -P "${DATA_DIR}"
done < ava_test.txt
