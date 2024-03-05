#!/bin/bash
set -e

./download.sh
./cut_15min_30min.sh
./extract_frames.sh
./download_annotations.sh

# etc.
