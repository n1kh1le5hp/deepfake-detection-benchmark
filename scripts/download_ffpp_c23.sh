#!/usr/bin/env bash
# Download + extract FaceForensics++ C23 from Kaggle mirror (xdxd003/ff-c23, ~16.7 GiB).
set -u
ROOT=/home/nikhi/id_test_reconstruction
VENV="$ROOT/.venv/bin"
DEST="$ROOT/data/raw/FaceForensics++"
mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "== downloading FF++ C23 from Kaggle (~16.7 GiB) =="
"$VENV/kaggle" datasets download -d xdxd003/ff-c23
rc=$?; echo "kaggle rc=$rc"
if [ "$rc" -ne 0 ]; then echo "DOWNLOAD FAILED"; exit 1; fi

ZIP=$(ls -1 *.zip 2>/dev/null | head -1)
echo "archive: $ZIP ($(du -h "$ZIP" | cut -f1))"

echo "== extracting (this is the slow part for ~17 GiB) =="
if command -v unzip >/dev/null 2>&1; then
  unzip -q -o "$ZIP"
else
  "$VENV/python" -c "import zipfile; zipfile.ZipFile('$ZIP').extractall('.')"
fi

echo "== category folders =="
find . -maxdepth 2 -type d | sort
echo "== DONE =="
