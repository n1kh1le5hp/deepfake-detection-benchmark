#!/usr/bin/env bash
# Resume the FF++ C23 download (curl -C -, Kaggle basic-auth), verify, extract.
set -u
ROOT=/home/nikhi/id_test_reconstruction
VENV="$ROOT/.venv/bin"
DEST="$ROOT/data/raw/FaceForensics++"
cd "$DEST" || exit 1
ZIP=ff-c23.zip
EXPECTED=17891522096
URL="https://www.kaggle.com/api/v1/datasets/download/xdxd003/ff-c23"

# read creds at runtime (not stored in this file)
creds=$("$VENV/python" -c "import json;d=json.load(open('/home/nikhi/.kaggle/kaggle.json'));print(d['username']+':'+d['key'])")

echo "== resuming FF++ download (curl -C -) =="
curl -C - -L -u "$creds" -o "$ZIP" "$URL"
rc=$?; echo "curl rc=$rc"

sz=$(stat -c %s "$ZIP")
echo "final size: $sz (expected $EXPECTED)"
if [ "$sz" -lt "$EXPECTED" ]; then echo "INCOMPLETE — re-run to resume"; exit 1; fi
echo "download complete ✓"

echo "== extracting (~16.7 GiB, may take a few min) =="
"$VENV/python" -c "import zipfile; zipfile.ZipFile('$ZIP').extractall('.')"

echo "== category folders =="
find . -maxdepth 2 -type d | sort
echo "== DONE =="
