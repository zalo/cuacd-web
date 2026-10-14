// Build the npm package with @deno/dnt (ESM + type declarations) into ./dist
// and write the root package.json that points at it. dist/ is committed so the
// package installs straight from GitHub:  npm install github:zalo/cuacd-web
//   deno run -A scripts/build_npm.ts [version]
import { build, emptyDir } from "@deno/dnt";

const version = Deno.args[0] ?? JSON.parse(await Deno.readTextFile("deno.json")).version ?? "0.1.0";
const WEBGPU_TYPES = "0.1.68";
const OUT = "./dist";

const pkg = {
  name: "cuacd-web",
  version,
  description: "Approximate convex decomposition on WebGPU: a TypeScript + WGSL port of CuACD / CoACD",
  license: "LGPL-2.1",
  repository: { type: "git", url: "git+https://github.com/zalo/cuacd-web.git" },
  homepage: "https://zalo.github.io/cuacd-web/",
  bugs: { url: "https://github.com/zalo/cuacd-web/issues" },
  keywords: ["convex decomposition", "webgpu", "coacd", "collision", "mesh", "three.js"],
  type: "module",
  sideEffects: false,
  engines: { node: ">=20" },
  // dnt adds earcut (from the import map) to dependencies
  dependencies: { "@webgpu/types": `^${WEBGPU_TYPES}` } as Record<string, string>,
  devDependencies: { "@types/earcut": "^3.0.0", "@types/node": "^20" },
};

// Build from a copy of the sources with a WebGPU types reference prepended:
// Deno ships WebGPU types, tsc (which dnt uses) needs @webgpu/types.
await emptyDir(OUT);
await emptyDir("./.npm-src");
function* walk(dir: string): Generator<string> {
  for (const e of Deno.readDirSync(dir)) {
    if (e.isDirectory) yield* walk(`${dir}/${e.name}`);
    else yield `${dir}/${e.name}`;
  }
}
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
  outDir: OUT,
  shims: {},
  test: false,
  typeCheck: false, // dnt's own import.meta polyfill fails tsc under @types/node; the sources are type-checked by Deno
  declaration: "separate",
  declarationMap: false,
  scriptModule: false, // ESM only: the WASM module and import.meta need it
  skipNpmInstall: true,
  compilerOptions: { lib: ["ES2022", "DOM", "DOM.Iterable"], target: "ES2022" },
  package: pkg,
  postBuild() {
    Deno.removeSync("./.npm-src", { recursive: true });
    // dnt's import.meta polyfill is needed at runtime (the Emscripten module
    // uses import.meta.url) but its declarations need @types/node, which
    // consumers should not have to install: drop the polyfill from the types only
    const dts = `${OUT}/types/src/mod.d.ts`;
    const lines = Deno.readTextFileSync(dts).split("\n").filter((l) => !l.includes("_dnt.polyfills"));
    Deno.writeTextFileSync(dts, `/// <reference types="@webgpu/types" />\n` + lines.join("\n"));
    for (const f of [`${OUT}/types/_dnt.polyfills.d.ts`, `${OUT}/types/_dnt.polyfills.d.ts.map`]) {
      try { Deno.removeSync(f); } catch { /* not emitted */ }
    }
    // no declaration maps: they would only point at a copy of src/
    Deno.removeSync(`${OUT}/src`, { recursive: true });
    for (const f of walk(`${OUT}/types`)) if (f.endsWith(".d.ts.map")) Deno.removeSync(f);
    // The package root is the repository root (so `npm install github:zalo/cuacd-web`
    // works): move dnt's package.json up with its paths rewritten to dist/.
    const generated = JSON.parse(Deno.readTextFileSync(`${OUT}/package.json`));
    Deno.removeSync(`${OUT}/package.json`);
    const root = {
      ...pkg,
      dependencies: generated.dependencies,
      module: "./dist/esm/src/mod.js",
      types: "./dist/types/src/mod.d.ts",
      exports: { ".": { types: "./dist/types/src/mod.d.ts", import: "./dist/esm/src/mod.js", default: "./dist/esm/src/mod.js" } },
      files: ["dist", "LICENSE", "README.md"],
      scripts: { test: "node scripts/test_npm.mjs" },
    };
    Deno.writeTextFileSync("package.json", JSON.stringify(root, null, 2) + "\n");
  },
});
