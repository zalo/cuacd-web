// Page smoke test without a GPU: serves demo/, loads it in headless Chromium
// (software WebGL; the worker reports that WebGPU is unavailable) and
// checks that the page, manifold-3d and the UI come up without JS errors.
// deno run --allow-all tests/demo_smoke.ts [chromium]
const chromium = Deno.args[0] ?? "chromium";
const port = 8123;
const server = new Deno.Command(Deno.execPath(), {
  args: ["run", "--allow-net", "--allow-read", "--allow-env", new URL("../demo/serve.ts", import.meta.url).pathname],
  env: { PORT: String(port) }, stdout: "null", stderr: "null",
}).spawn();
await new Promise((r) => setTimeout(r, 800));
const probe = `
<script type="module">
  const log = [];
  for (const k of ["log", "warn", "error"]) { const o = console[k].bind(console); console[k] = (...a) => { log.push(k + ": " + a.map(String).join(" ")); o(...a); }; }
  addEventListener("error", (e) => log.push("error: " + e.message));
  addEventListener("unhandledrejection", (e) => log.push("rejection: " + (e.reason?.message ?? e.reason)));
  setTimeout(() => {
    const msg = document.getElementById("msg")?.textContent, sub = document.getElementById("sub")?.textContent;
    const gui = document.querySelectorAll(".lil-gui .controller").length;
    document.title = JSON.stringify({ msg, sub, gui, log });
  }, 9000);
</script>`;
// serve a probe page that embeds the demo page's markup plus the hooks
const html = (await Deno.readTextFile(new URL("../demo/index.html", import.meta.url))).replace("</head>", probe + "</head>");
await Deno.writeTextFile(new URL("../demo/_smoke.html", import.meta.url), html);
try {
  const out = await new Deno.Command(chromium, {
    args: ["--headless=new", "--no-sandbox", "--ozone-platform=headless", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--virtual-time-budget=15000", "--dump-dom", `http://localhost:${port}/_smoke.html`],
    stdout: "piped", stderr: "null",
  }).output();
  const dom = new TextDecoder().decode(out.stdout);
  const m = dom.match(/<title>(.*?)<\/title>/s);
  const title = m ? m[1].replace(/&quot;/g, '"').replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">") : "";
  let r: { msg: string; sub: string; gui: number; log: string[] } | null = null;
  try { r = JSON.parse(title); } catch { /* not our title */ }
  if (!r) { console.log("smoke: no probe result; dom head:", dom.slice(0, 300)); Deno.exit(1); }
  console.log(`status: ${r.msg} | ${r.sub}`);
  console.log(`gui controllers: ${r.gui}`);
  for (const l of r.log) console.log("  " + l);
  // a headless browser without WebGL logs context-creation errors from three.js; the app runs without the view
  const bad = r.log.filter((l) => (l.startsWith("error") || l.startsWith("rejection")) && !/WebGL context/.test(l));
  const ok = r.gui > 10 && bad.length === 0;
  console.log(ok ? "SMOKE: OK" : "SMOKE: FAIL");
  Deno.exit(ok ? 0 : 1);
} finally {
  await Deno.remove(new URL("../demo/_smoke.html", import.meta.url)).catch(() => {});
  server.kill();
}
