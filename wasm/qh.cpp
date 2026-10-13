// qh.cpp — Emscripten wrapper around Antti Kuukka's quickhull (public domain).
// Exposes a tiny C API: upload points, compute, read vertices/triangles.
#include "quickhull/QuickHull.hpp"
#include <emscripten/emscripten.h>
#include <cstdlib>
#include <vector>

static std::vector<double> g_verts;
static std::vector<unsigned> g_tris;

extern "C" {
EMSCRIPTEN_KEEPALIVE double* qh_alloc(int nDoubles) { return (double*)std::malloc(sizeof(double) * (size_t)nDoubles); }
EMSCRIPTEN_KEEPALIVE void qh_free(void* p) { std::free(p); }

// Returns the triangle count. Vertices are compacted to hull vertices.
EMSCRIPTEN_KEEPALIVE int qh_compute(const double* pts, int n, double eps) {
  quickhull::QuickHull<double> qh;
  auto hull = qh.getConvexHull(pts, (size_t)n, true, false, eps);
  const auto& ib = hull.getIndexBuffer();
  const auto& vb = hull.getVertexBuffer();
  g_verts.resize(vb.size() * 3);
  for (size_t i = 0; i < vb.size(); i++) { g_verts[3*i] = vb[i].x; g_verts[3*i+1] = vb[i].y; g_verts[3*i+2] = vb[i].z; }
  g_tris.assign(ib.begin(), ib.end());
  return (int)(g_tris.size() / 3);
}
EMSCRIPTEN_KEEPALIVE int qh_nv() { return (int)(g_verts.size() / 3); }
EMSCRIPTEN_KEEPALIVE double* qh_verts() { return g_verts.data(); }
EMSCRIPTEN_KEEPALIVE unsigned* qh_tris() { return g_tris.data(); }
}
