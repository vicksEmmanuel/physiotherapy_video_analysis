#!/bin/bash
set -e

./download_win.sh
./cut_15min_30min_win.sh
./extract_frames.sh
./download_annotations.sh

# etc.
