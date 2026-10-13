// Build the npm package in ./npm with @deno/dnt (ESM + type declarations).
//   deno run -A scripts/build_npm.ts [version]
import { build, emptyDir } from "@deno/dnt";

const version = Deno.args[0] ?? "0.1.0";
const WEBGPU_TYPES = "0.1.68";

// Build from a copy of the sources with a WebGPU types reference prepended:
// Deno ships WebGPU types, tsc (which dnt uses) needs @webgpu/types.
await emptyDir("./npm");
await emptyDir("./.npm-src");
const copyDir = async (from: string, to: string) => {
  await Deno.mkdir(to, { recursive: true });
  for await (const e of Deno.readDir(from)) {
    if (e.isDirectory) await copyDir(`${from}/${e.name}`, `${to}/${e.name}`);
    else await Deno.copyFile(`${from}/${e.name}`, `${to}/${e.name}`);
  }
};
await copyDir("./src", "./.npm-src/src");
await copyDir("./wasm", "./.npm-src/wasm");
const entry = "./.npm-src/src/mod.ts";
await Deno.writeTextFile(entry, `/// <reference types="@webgpu/types" />\n` + await Deno.readTextFile(entry));

await build({
  entryPoints: [entry],
  outDir: "./npm",
  shims: {},
  test: false,
  typeCheck: false, // dnt's own import.meta polyfill fails tsc under @types/node; the sources are type-checked by Deno
  declaration: "separate",
  scriptModule: false, // ESM only: the WASM module and import.meta need it
  compilerOptions: { lib: ["ES2022", "DOM", "DOM.Iterable"], target: "ES2022" },
  package: {
    name: "cuacd-web",
    version,
    description: "Approximate convex decomposition on WebGPU: a TypeScript + WGSL port of CuACD / CoACD",
    license: "LGPL-2.1",
    repository: { type: "git", url: "git+https://github.com/zalo/cuacd-web.git" },
    homepage: "https://zalo.github.io/cuacd-web/",
    bugs: { url: "https://github.com/zalo/cuacd-web/issues" },
    keywords: ["convex decomposition", "webgpu", "coacd", "collision", "mesh", "three.js"],
    sideEffects: false,
    engines: { node: ">=20" },
    dependencies: { "@webgpu/types": `^${WEBGPU_TYPES}` },
    devDependencies: { "@types/earcut": "^3.0.0", "@types/node": "^20" },
  },
  postBuild() {
    Deno.copyFileSync("LICENSE", "npm/LICENSE");
    Deno.copyFileSync("README.md", "npm/README.md");
    Deno.removeSync("./.npm-src", { recursive: true });
    // dnt's import.meta polyfill is needed at runtime (the Emscripten module
    // uses import.meta.url) but its declarations need @types/node, which
    // consumers should not have to install: drop the polyfill from the types only
    const dtsEntry = "npm/types/src/mod.d.ts";
    Deno.writeTextFileSync(dtsEntry, Deno.readTextFileSync(dtsEntry).split("\n").filter((l) => !l.includes("_dnt.polyfills")).join("\n"));
    for (const f of ["npm/types/_dnt.polyfills.d.ts", "npm/types/_dnt.polyfills.d.ts.map"]) {
      try { Deno.removeSync(f); } catch { /* not emitted */ }
    }
    // consumers need the WebGPU types reference in the declarations
    const dts = "npm/types/src/mod.d.ts";
    Deno.writeTextFileSync(dts, `/// <reference types="@webgpu/types" />\n` + Deno.readTextFileSync(dts));
  },
});
