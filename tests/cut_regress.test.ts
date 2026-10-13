// Opt-in regression cases captured from real decompositions (/tmp/meshes/fail_*.obj + .json).
import { assert } from "@std/assert";
import { isClosed, parseOBJ, volume } from "../src/mesh.ts";
import { planeCut, type Plane } from "../src/cpu/cut.ts";

const cases: string[] = [];
try { for (const e of Deno.readDirSync("/tmp/meshes")) if (e.name.startsWith("fail") && e.name.endsWith(".obj")) cases.push("/tmp/meshes/" + e.name); } catch { /* none */ }

Deno.test({ name: "planeCut: captured failure cases cut into closed halves", ignore: cases.length === 0, fn: async () => {
  for (const f of cases) {
    const mesh = parseOBJ(await Deno.readTextFile(f));
    let plane: Plane;
    try { plane = JSON.parse(await Deno.readTextFile(f.replace(".obj", ".json"))); } catch { continue; }
    const r = planeCut(mesh, plane);
    assert(r.ok, `${f}: ${r.reason}`);
    assert(isClosed(r.pos!), `${f}: pos not closed`);
    assert(isClosed(r.neg!), `${f}: neg not closed`);
    const vin = volume(mesh);
    assert(Math.abs(volume(r.pos!) + volume(r.neg!) - vin) < 1e-4 * vin, `${f}: volume mismatch`);
    console.log(`${f}: ok`);
  }
}});
