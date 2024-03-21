#!/bin/bash
set -e

./cut_15min_30min_win.sh
./extract_frames.sh
./download_annotations.sh

# etc.
