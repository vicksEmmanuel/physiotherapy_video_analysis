#!/bin/bash
DATA_DIR="videos"

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "${DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${DATA_DIR}
fi

# wget https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt

for line in $(cat ava_test.txt)
do
  $cleanedLine  = $(echo "$line" | sed 's/%0D//g')
  wget https://s3.amazonaws.com/ava-dataset/trainval/$line -P ${DATA_DIR}
done