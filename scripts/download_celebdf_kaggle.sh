#!/usr/bin/env bash
# Download + extract Celeb-DF v2 from the Kaggle mirror (reubensuju/celeb-df-v2).
# Requires ~/.kaggle/kaggle.json (already placed, chmod 600).
set -u
ROOT=/home/nikhi/id_test_reconstruction
VENV="$ROOT/.venv/bin"
DEST="$ROOT/data/raw/Celeb-DF"
mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "== downloading Celeb-DF v2 from Kaggle (~9.9 GiB) =="
"$VENV/kaggle" datasets download -d reubensuju/celeb-df-v2
rc=$?; echo "kaggle download rc=$rc"
if [ "$rc" -ne 0 ]; then echo "DOWNLOAD FAILED"; exit 1; fi

ZIP=$(ls -1 *.zip 2>/dev/null | head -1)
echo "downloaded archive: $ZIP ($(du -h "$ZIP" | cut -f1))"

echo "== extracting =="
if command -v unzip >/dev/null 2>&1; then
  unzip -q -o "$ZIP"
else
  "$VENV/python" -c "import zipfile; zipfile.ZipFile('$ZIP').extractall('.')"
fi

echo "== structure & counts =="
find . -maxdepth 1 -type d | sort
for d in Celeb-real Celeb-synthesis YouTube-real; do
  echo "  $d: $(find "$d" -type f -name '*.mp4' 2>/dev/null | wc -l) mp4"
done
echo "total mp4: $(find . -type f -name '*.mp4' | wc -l)"
echo "== DONE =="
