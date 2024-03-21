@echo off
setlocal EnableDelayedExpansion

set DATA_DIR=videos

if not exist "%DATA_DIR%" (
    echo "%DATA_DIR% doesn't exist. Creating it."
    mkdir "%DATA_DIR%"
)

bitsadmin /transfer downloadJob /download /priority normal https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt "%CD%\ava_file_names_trainval_v2.1.txt"

for /F "delims=" %%line in (ava_file_names_trainval_v2.1.txt) do (
    bitsadmin /transfer downloadJob /download /priority normal https://s3.amazonaws.com/ava-dataset/trainval/%%line "%CD%\%DATA_DIR%\%%~nxline"
)

echo "Downloads complete."
endlocal