#!/usr/bin/env bash
# Download + verify + extract ForgeryNet Video Test Set, Training List, Validation List.
set -u
ROOT=/home/nikhi/id_test_reconstruction
VENV="$ROOT/.venv/bin"
BASE="$ROOT/data/raw/ForgeryNet"
VROOT="$BASE/videos"
LROOT="$BASE/lists"
mkdir -p "$VROOT" "$LROOT"

VID_ID=1CSJOkDR_jJvq7qUGP8oGcwpoPdKGzfgb
VID_MD5=92b870009cb03832b2c2795b6a35629f
TRAIN_LIST_ID=1ZqPqmYdqmq4_HLZu1c5ySXhcR1WQWQ9L
VAL_LIST_ID=14Rqq4C4oHK6FEqyTIiVcQy7wQuMHGQzu

echo "== [1/3] Video Test Set =="
cd "$VROOT"
"$VENV/gdown" "$VID_ID" -O public_test_videos.tar
rc=$?; echo "gdown(video) rc=$rc"
if [ "$rc" -ne 0 ]; then echo "VIDEO DOWNLOAD FAILED"; exit 1; fi
actual=$(md5sum public_test_videos.tar | awk '{print $1}')
echo "video md5 actual=$actual expected=$VID_MD5"
if [ "$actual" != "$VID_MD5" ]; then echo "VIDEO MD5 MISMATCH"; exit 2; fi
echo "video MD5 OK; extracting..."
tar xf public_test_videos.tar
echo "video files extracted: $(find . -type f -not -name '*.tar' | wc -l)"

echo "== [2/3] Training List =="
cd "$LROOT"
"$VENV/gdown" "$TRAIN_LIST_ID" -O training_list
echo "training_list type: $(file -b training_list 2>/dev/null | head -c 80)"
if tar tf training_list >/dev/null 2>&1; then echo "  -> is a tar, extracting"; mkdir -p training_list_x; tar xf training_list -C training_list_x; find training_list_x -type f | head; else echo "  -> not a tar (likely json)"; fi
ls -lh training_list

echo "== [3/3] Validation List =="
"$VENV/gdown" "$VAL_LIST_ID" -O validation_list
echo "validation_list type: $(file -b validation_list 2>/dev/null | head -c 80)"
if tar tf validation_list >/dev/null 2>&1; then echo "  -> is a tar, extracting"; mkdir -p validation_list_x; tar xf validation_list -C validation_list_x; find validation_list_x -type f | head; else echo "  -> not a tar (likely json)"; fi
ls -lh validation_list

echo "== DONE =="
