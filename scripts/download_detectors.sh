#!/usr/bin/env bash
# Download pretrained deepfake-detector weights for all available methods.
# Sources verified by research agents (June 2026). Continues on per-method failure.
set -u
ROOT=/home/nikhi/id_test_reconstruction
VENV="$ROOT/.venv/bin"
W="$ROOT/external/weights"
mkdir -p "$W"/{xception,meso4,mesoinception,ffd,lrnet,facexray,multi_attention,patch,meso4Incep}
cd "$W" || exit 1

ok(){ echo "  [OK]   $1 -> $(du -h "$2" 2>/dev/null | cut -f1)"; }
fail(){ echo "  [FAIL] $1 ($2)"; }

echo "### 1. Xception (DeepfakeBench release)"
curl -sL -o xception/xception_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth" \
  && [ -s xception/xception_best.pth ] && ok "xception" "xception/xception_best.pth" || fail "xception" "curl"

echo "### 2. FFD (DeepfakeBench release)"
curl -sL -o ffd/ffd_best.pth "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/ffd_best.pth" \
  && [ -s ffd/ffd_best.pth ] && ok "ffd" "ffd/ffd_best.pth" || fail "ffd" "curl"

echo "### 3. MesoNet-4 (DariusAf/MesoNet)"
curl -sL -o meso4/Meso4_DF.h5 "https://raw.githubusercontent.com/DariusAf/MesoNet/master/weights/Meso4_DF.h5" \
  && [ -s meso4/Meso4_DF.h5 ] && ok "meso4" "meso4/Meso4_DF.h5" || fail "meso4" "curl"

echo "### 4. MesoInception-4 (DariusAf/MesoNet)"
curl -sL -o meso4Incep/MesoInception_DF.h5 "https://raw.githubusercontent.com/DariusAf/MesoNet/master/weights/MesoInception_DF.h5" \
  && [ -s meso4Incep/MesoInception_DF.h5 ] && ok "mesoinception" "meso4Incep/MesoInception_DF.h5" || fail "mesoinception" "curl"

echo "### 5. LRNet (frederickszk/LRNet)"
curl -sL -o lrnet/g1.pth "https://raw.githubusercontent.com/frederickszk/LRNet/main/training/weights/torch/g1.pth" && ok "lrnet-g1" "lrnet/g1.pth" || fail "lrnet-g1" "curl"
curl -sL -o lrnet/g2.pth "https://raw.githubusercontent.com/frederickszk/LRNet/main/training/weights/torch/g2.pth" && ok "lrnet-g2" "lrnet/g2.pth" || fail "lrnet-g2" "curl"

echo "### 6. Face X-ray (wkq-wukaiqi, Google Drive)"
"$VENV/gdown" 1Vb2sCfeEdQejSSPjlBwNyeXiUMCTOR-H -O facexray/best_model.pth.tar \
  && [ -s facexray/best_model.pth.tar ] && ok "facexray" "facexray/best_model.pth.tar" || fail "facexray" "gdown"

echo "### 7. Multi-attention (yoctta, Google Drive)"
"$VENV/gdown" 1lYyUe99Goh1YCilt1IOiD9oMO6ig8j1o -O multi_attention/multi_attention.pth \
  && [ -s multi_attention/multi_attention.pth ] && ok "multi_attention" "multi_attention/multi_attention.pth" || fail "multi_attention" "gdown"

echo "### 8. Patch-forensics Xception (chail, Google Drive folder)"
"$VENV/gdown" --folder "https://drive.google.com/drive/folders/1_LekvsBFE2T9N3Wikkll3xjlogI-cSoH" -O patch/ 2>&1 | tail -4 \
  && ok "patch (folder)" "patch/" || fail "patch" "gdown folder"

echo
echo "### SUMMARY ###"
find "$W" -type f -printf '%p\t%s\n' | awk -F'\t' '{printf "%-60s %8.1f MB\n",$1,$2/1048576}'
