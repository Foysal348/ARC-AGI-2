#!/usr/bin/env bash
# E23 P2: install the Merlin 3.0.0 engine (vLLM 0.32.0 fork) OFFLINE on the pinned Kaggle image (Ubuntu 22.04,
# glibc 2.35, Python 3.12, GCC 11) from the P2 payload. Runs Merlin's own release/install-env.sh unchanged:
#   - CPython 3.13 from the payload (uv-managed install, fixed path /tmp/merlin_uvpy), no Python download;
#   - every locked dependency from the payload wheelhouse (uv.toml: no-index + find-links), UV_OFFLINE=1;
#   - GCC 13 from conda-forge (payload, fixed path /tmp/merlin_gcc13) for the installer's GCC >= 13 rule;
#   - the release wheel through MERLIN_WHEEL (precompiled products, no native build).
# Then the glibc 2.35 fix (glibc235_fix.py: weak GLIBC_2.38 need + LD_PRELOAD shim), the build-product hashes
# re-recorded for the patched files, and install-env.sh --verify.
#   bash install_merlin_offline.sh <payload-dir> <dest-dir>
# Prints MERLIN_VENV= / MERLIN_TREE= / MERLIN_SHIM= / MERLIN_CC= / MERLIN_CXX= lines and INSTALL OK at the end.
set -euo pipefail
PAY="$(cd "${1:?usage: install_merlin_offline.sh <payload-dir> <dest-dir>}" && pwd -P)"
DEST="${2:?usage: install_merlin_offline.sh <payload-dir> <dest-dir>}"
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd -P)"
die() { echo "install_merlin_offline: FAIL: $*" >&2; exit 1; }
WHEEL="$(ls "$PAY"/vllm-0.32.0+merlin.*-cp313-cp313-linux_x86_64.whl 2>/dev/null | head -1)"
[ -f "$WHEEL" ] || die "no Merlin wheel in $PAY"
for f in merlin-engine.tar merlin_uvpy.tar.gz merlin_gcc13.tar.gz glibc235_fix.py rehash_products.py libglibc238_shim.so; do
  [ -e "$PAY/$f" ] || die "payload is missing $f"
done
[ -d "$PAY/wheelhouse" ] || die "payload is missing wheelhouse/"

# fixed-prefix trees (the conda env and the uv Python record their own install prefix)
[ -x /tmp/merlin_uvpy/.ok ] || { rm -rf /tmp/merlin_uvpy; tar -xzf "$PAY/merlin_uvpy.tar.gz" -C /tmp; touch /tmp/merlin_uvpy/.ok; chmod +x /tmp/merlin_uvpy/.ok; }
[ -x /tmp/merlin_gcc13/.ok ] || { rm -rf /tmp/merlin_gcc13; tar -xzf "$PAY/merlin_gcc13.tar.gz" -C /tmp; touch /tmp/merlin_gcc13/.ok; chmod +x /tmp/merlin_gcc13/.ok; }
CC13="$(ls /tmp/merlin_gcc13/bin/x86_64-conda-linux-gnu-gcc 2>/dev/null || true)"
CXX13="$(ls /tmp/merlin_gcc13/bin/x86_64-conda-linux-gnu-g++ 2>/dev/null || true)"
[ -x "$CC13" ] && [ -x "$CXX13" ] || die "GCC 13 not found under /tmp/merlin_gcc13/bin"

TREE="$DEST/merlin-engine"
VENV="$DEST/venv"
[ ! -e "$VENV" ] || die "$VENV already exists"
rm -rf "$TREE"
tar -xf "$PAY/merlin-engine.tar" -C "$DEST"
[ -f "$TREE/release/install-env.sh" ] || die "merlin-engine.tar did not give $TREE"
git config --global --add safe.directory "$TREE" 2>/dev/null || true

cat > "$DEST/uv-offline.toml" <<EOF
[pip]
no-index = true
find-links = ["$PAY/wheelhouse"]
EOF
export UV_CONFIG_FILE="$DEST/uv-offline.toml" UV_OFFLINE=1 UV_PYTHON_DOWNLOADS=never
export UV_PYTHON_INSTALL_DIR=/tmp/merlin_uvpy UV_CACHE_DIR="$DEST/uv-cache" UV_LINK_MODE=copy
export CC="$CC13" CXX="$CXX13" CUDAHOSTCXX="$CXX13"
unset CUDA_HOME CUDA_PATH PYTHONPATH PYTHONHOME VLLM_USE_PRECOMPILED VLLM_PRECOMPILED_WHEEL_LOCATION CMAKE_ARGS || true
export MERLIN_WHEEL="$WHEEL"
(cd "$TREE" && bash release/install-env.sh "$VENV")
unset MERLIN_WHEEL

mapfile -t SOS < <(find "$TREE/vllm" -type f \( -name '*.so' -o -name '*.so.*' \))
python3 -I "$PAY/glibc235_fix.py" "${SOS[@]}"
python3 -I "$PAY/rehash_products.py" "$TREE"
cp -f "$PAY/libglibc238_shim.so" "$DEST/libglibc238_shim.so"
(cd "$TREE" && bash release/install-env.sh --verify "$VENV")

echo "MERLIN_VENV=$VENV"
echo "MERLIN_TREE=$TREE"
echo "MERLIN_SHIM=$DEST/libglibc238_shim.so"
echo "MERLIN_CC=$CC13"
echo "MERLIN_CXX=$CXX13"
echo "INSTALL OK"
