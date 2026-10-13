// Static server for the demo: deno task serve → http://localhost:8000/
// (run `deno task bundle` first; the demo imports ./cuacd.bundle.js)
const root = new URL("./", import.meta.url).pathname;
const types: Record<string, string> = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
  ".obj": "text/plain", ".stl": "application/octet-stream", ".gz": "application/gzip", ".glb": "model/gltf-binary",
  ".ply": "application/octet-stream", ".json": "application/json", ".css": "text/css",
};
const port = Number(Deno.env.get("PORT") ?? 8000);
Deno.serve({ port }, async (req) => {
  let path = decodeURIComponent(new URL(req.url).pathname);
  if (path === "/") path = "/index.html";
  if (path.includes("..")) return new Response("forbidden", { status: 403 });
  try {
    const data = await Deno.readFile(root + path);
    const ext = path.slice(path.lastIndexOf("."));
    return new Response(data, { headers: { "content-type": types[ext] ?? "application/octet-stream", "cache-control": "no-store" } });
  } catch {
    return new Response("not found", { status: 404 });
  }
});
