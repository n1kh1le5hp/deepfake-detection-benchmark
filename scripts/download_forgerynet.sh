#!/usr/bin/env bash
# Download + verify + extract the ForgeryNet Image Test Set from Google Drive.
# Resumable-safe: re-running won't clobber a good tar (md5 checked before extract).
set -u
ROOT=/home/nikhi/id_test_reconstruction
VENV="$ROOT/.venv/bin"
DIR="$ROOT/data/raw/ForgeryNet/imgs"
IMG_ID=1conYQXWguAwJ1eEwewHyMBGUtgjgR_sM
EXPECTED_MD5=e4218faa4b934345977202edfbf89301
TAR=public_test_images.tar

mkdir -p "$DIR"
cd "$DIR" || exit 1

echo "== gdown -> $TAR (id=$IMG_ID) =="
"$VENV/gdown" "$IMG_ID" -O "$TAR"
rc=$?
echo "gdown exit code: $rc"
if [ "$rc" -ne 0 ]; then echo "DOWNLOAD FAILED (rc=$rc)"; exit "$rc"; fi

echo "== verify md5 =="
actual=$(md5sum "$TAR" | awk '{print $1}')
echo "actual:   $actual"
echo "expected: $EXPECTED_MD5"
if [ "$actual" != "$EXPECTED_MD5" ]; then echo "MD5 MISMATCH — keeping tar for retry"; exit 2; fi
echo "MD5 OK"

echo "== extracting =="
tar xf "$TAR"
echo "extracted files: $(find . -type f -not -name "$TAR" | wc -l)"
echo "== DONE =="
