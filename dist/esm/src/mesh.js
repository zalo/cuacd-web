// mesh.ts — plain triangle-mesh helpers shared by the CPU and GPU paths.
//
// A Mesh is a flat xyz float array plus a flat triangle-index array. All
// geometry handed to the decomposer is expected to be closed and outward
// wound; `isClosed` checks the former, `signedVolume` reveals the latter.
export function nv(m) {
    return m.verts.length / 3;
}
export function nt(m) {
    return m.tris.length / 3;
}
export function makeBox(lo, hi) {
    const [x0, y0, z0] = lo;
    const [x1, y1, z1] = hi;
    const verts = new Float32Array([
        x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0,
        x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
    ]);
    const tris = new Uint32Array([
        0, 2, 1, 0, 3, 2, // bottom
        4, 5, 6, 4, 6, 7, // top
        0, 1, 5, 0, 5, 4, // front
        2, 3, 7, 2, 7, 6, // back
        1, 2, 6, 1, 6, 5, // right
        0, 4, 7, 0, 7, 3, // left
    ]);
    return { verts, tris };
}
export function mergeMeshes(a, b) {
    const verts = new Float32Array(a.verts.length + b.verts.length);
    verts.set(a.verts);
    verts.set(b.verts, a.verts.length);
    const tris = new Uint32Array(a.tris.length + b.tris.length);
    tris.set(a.tris);
    const off = nv(a);
    for (let i = 0; i < b.tris.length; i++)
        tris[a.tris.length + i] = b.tris[i] + off;
    return { verts, tris };
}
/** Unit cube [0,1]^3 (matches tests/conftest.py unit_cube). */
export function unitCube() {
    return makeBox([0, 0, 0], [1, 1, 1]);
}
/** L-shape from two boxes joined along x (matches tests/conftest.py l_shape). */
export function lShape() {
    return mergeMeshes(makeBox([0, 0, 0], [1, 1, 1]), makeBox([1, 0, 0], [2, 1, 0.5]));
}
/** Signed volume via the divergence theorem, anchored at vertex 0. */
export function signedVolume(m) {
    const v = m.verts, t = m.tris;
    if (t.length === 0)
        return 0;
    const ox = v[0], oy = v[1], oz = v[2];
    let s = 0;
    for (let i = 0; i < t.length; i += 3) {
        const a = t[i] * 3, b = t[i + 1] * 3, c = t[i + 2] * 3;
        const ax = v[a] - ox, ay = v[a + 1] - oy, az = v[a + 2] - oz;
        const bx = v[b] - ox, by = v[b + 1] - oy, bz = v[b + 2] - oz;
        const cx = v[c] - ox, cy = v[c + 1] - oy, cz = v[c + 2] - oz;
        s += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
    }
    return s / 6;
}
export function volume(m) {
    return Math.abs(signedVolume(m));
}
/** True when every directed edge has exactly one reverse twin. */
/**
 * Directed edges in CSR form by source vertex: edge e = (src, dst[e]) of
 * triangle face[e]; edges of vertex v are dst[off[v] .. off[v+1]).
 */
export function edgeCSR(m) {
    const T = m.tris;
    const n = nv(m);
    const off = new Uint32Array(n + 1);
    for (let i = 0; i < T.length; i++)
        off[T[i] + 1]++;
    for (let v = 0; v < n; v++)
        off[v + 1] += off[v];
    const fill = off.slice(0, n);
    const dst = new Uint32Array(T.length);
    const face = new Uint32Array(T.length);
    for (let i = 0; i < T.length; i += 3) {
        for (let e = 0; e < 3; e++) {
            const a = T[i + e], b = T[i + ((e + 1) % 3)];
            const k = fill[a]++;
            dst[k] = b;
            face[k] = i / 3;
        }
    }
    return { off, dst, face };
}
/** Closed in the edge-twin sense: every directed edge occurs once and its reverse occurs once. */
export function isClosed(m) {
    const { off, dst } = edgeCSR(m);
    const n = off.length - 1;
    const count = (a, b) => {
        let c = 0;
        for (let k = off[a]; k < off[a + 1]; k++)
            if (dst[k] === b)
                c++;
        return c;
    };
    for (let a = 0; a < n; a++) {
        for (let k = off[a]; k < off[a + 1]; k++) {
            const b = dst[k];
            if (b >= n || count(a, b) !== 1 || count(b, a) !== 1)
                return false;
        }
    }
    return true;
}
export function bbox(m) {
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    const v = m.verts;
    for (let i = 0; i < v.length; i += 3) {
        for (let a = 0; a < 3; a++) {
            const x = v[i + a];
            if (x < lo[a])
                lo[a] = x;
            if (x > hi[a])
                hi[a] = x;
        }
    }
    return { lo, hi };
}
/** Center the mesh and scale its largest extent to 2 (CLI convention). */
export function normalize(m) {
    const { lo, hi } = bbox(m);
    const center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    const scale = extent > 0 ? extent / 2 : 1;
    const verts = new Float32Array(m.verts.length);
    for (let i = 0; i < verts.length; i += 3) {
        verts[i] = (m.verts[i] - center[0]) / scale;
        verts[i + 1] = (m.verts[i + 1] - center[1]) / scale;
        verts[i + 2] = (m.verts[i + 2] - center[2]) / scale;
    }
    return { mesh: { verts, tris: m.tris.slice() }, norm: { scale, center } };
}
export function denormalize(m, norm) {
    const verts = new Float32Array(m.verts.length);
    for (let i = 0; i < verts.length; i += 3) {
        verts[i] = m.verts[i] * norm.scale + norm.center[0];
        verts[i + 1] = m.verts[i + 1] * norm.scale + norm.center[1];
        verts[i + 2] = m.verts[i + 2] * norm.scale + norm.center[2];
    }
    return { verts, tris: m.tris };
}
/** Parse a binary or ASCII STL. Vertices are welded by exact position. */
export function parseSTL(buf) {
    const u8 = new Uint8Array(buf);
    const head = new TextDecoder().decode(u8.subarray(0, Math.min(80, u8.length)));
    const dv = new DataView(buf);
    const isBinary = !(head.trimStart().startsWith("solid") && !looksBinary(buf));
    const pos = [];
    if (isBinary) {
        const n = dv.getUint32(80, true);
        let off = 84;
        for (let i = 0; i < n; i++) {
            off += 12; // normal
            for (let k = 0; k < 9; k++)
                pos.push(dv.getFloat32(off + k * 4, true));
            off += 36 + 2;
        }
    }
    else {
        const text = new TextDecoder().decode(u8);
        const re = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g;
        let mm;
        while ((mm = re.exec(text)) !== null)
            pos.push(+mm[1], +mm[2], +mm[3]);
    }
    return weld(pos);
}
function looksBinary(buf) {
    if (buf.byteLength < 84)
        return false;
    const n = new DataView(buf).getUint32(80, true);
    return 84 + n * 50 === buf.byteLength;
}
/** Weld a triangle soup (9 floats per triangle) into an indexed mesh. */
export function weld(pos) {
    const map = new Map();
    const verts = [];
    const tris = [];
    for (let i = 0; i < pos.length; i += 3) {
        const x = pos[i], y = pos[i + 1], z = pos[i + 2];
        const k = `${x},${y},${z}`;
        let idx = map.get(k);
        if (idx === undefined) {
            idx = verts.length / 3;
            map.set(k, idx);
            verts.push(x, y, z);
        }
        tris.push(idx);
    }
    return { verts: new Float32Array(verts), tris: new Uint32Array(tris) };
}
/** Parse a simple OBJ (v / f lines, polygons fan-triangulated). */
export function parseOBJ(text) {
    const verts = [];
    const tris = [];
    for (const line of text.split("\n")) {
        if (line.startsWith("v ")) {
            const p = line.split(/\s+/);
            verts.push(+p[1], +p[2], +p[3]);
        }
        else if (line.startsWith("f ")) {
            const p = line.trim().split(/\s+/).slice(1).map((s) => parseInt(s.split("/")[0], 10) - 1);
            for (let i = 1; i + 1 < p.length; i++)
                tris.push(p[0], p[i], p[i + 1]);
        }
    }
    return { verts: new Float32Array(verts), tris: new Uint32Array(tris) };
}
export function toOBJ(parts) {
    const out = [];
    let base = 1;
    parts.forEach((m, i) => {
        out.push(`o part_${i}`);
        for (let k = 0; k < m.verts.length; k += 3)
            out.push(`v ${m.verts[k]} ${m.verts[k + 1]} ${m.verts[k + 2]}`);
        for (let k = 0; k < m.tris.length; k += 3) {
            out.push(`f ${m.tris[k] + base} ${m.tris[k + 1] + base} ${m.tris[k + 2] + base}`);
        }
        base += nv(m);
    });
    return out.join("\n") + "\n";
}
