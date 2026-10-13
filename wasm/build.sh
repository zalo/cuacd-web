#!/usr/bin/env bash
# Build wasm/qh.js (single-file ES module embedding the wasm) with Emscripten.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v em++ >/dev/null 2>&1; then
  if [ -f "$HOME/.local/emsdk/emsdk_env.sh" ]; then source "$HOME/.local/emsdk/emsdk_env.sh" >/dev/null 2>&1; fi
fi
em++ -O3 -std=c++17 -I. qh.cpp quickhull/QuickHull.cpp -o qh.js \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sSINGLE_FILE=1 -sALLOW_MEMORY_GROWTH=1 \
  -sENVIRONMENT=web,worker,node -sFILESYSTEM=0 \
  -sEXPORTED_FUNCTIONS=_qh_alloc,_qh_free,_qh_compute,_qh_nv,_qh_verts,_qh_tris \
  -sEXPORTED_RUNTIME_METHODS=HEAPF64,HEAPU32
echo "built $(pwd)/qh.js ($(stat -c %s qh.js) bytes)"
