/* DEFT Mission Control — vanilla JS, canvas scatter, no build step. */
"use strict";

const state = {
  summary: null, points: [], margins: [],
  iter: null,              // selected iteration label
  colorBy: "status_at_iter",
  dimUnused: true,
  facets: {},              // field -> Set(selected values)
  search: "",
  hover: null, pinned: null,
  view: { x: 0, y: 0, k: 1 },   // pan/zoom transform
  highlight: new Set(),          // agent evidence point ids
  transcript: [],                // for report export
  space: "siglip",               // embedding space for layout + hit-testing
  light: null,                   // lighting condition to view (null = channel 0)
  showEdges: false,              // draw k-NN retrieval edges for the iteration
  showStacks: false,             // badge overlapping points with a ×N count
  spider: null,                  // {members:[idx...]} — a fanned-out overlap cluster
  edgesCache: {},                // iter label -> edges payload
  whatIfThr: null,               // what-if threshold override (null = operating point)
  nnSameLabel: null,             // NN label-filter what-if (null = follow the run's spec)
  pulseT0: 0,                    // iteration-entry animation clock
};

function effThreshold(it) { return state.whatIfThr ?? it.threshold; }

function kpiConfusion(iterLabel, thr) {
  let fp = 0, fn = 0, nPass = 0, nNp = 0;
  for (const p of state.points) {
    if (p.kind !== "kpi") continue;
    const s = p.scores?.[iterLabel];
    if (s === undefined) continue;
    const predNP = s > thr + 1e-6;
    if (p.label === "PASS") { nPass++; if (predNP) fp++; }
    else { nNp++; if (!predNP) fn++; }
  }
  return { fp, fn, nPass, nNp, far: nPass ? 100 * fp / nPass : 0, recall: nNp ? 100 * (nNp - fn) / nNp : 0 };
}

const PALETTE = ["#76b900","#4e9cf5","#e5533c","#e5b83c","#b06ee5","#3cc8c8","#e57ab0","#9aa5b1","#5ce58a","#c8b43c"];
const cv = document.getElementById("map"), ctx = cv.getContext("2d");
const tooltip = document.getElementById("tooltip");

// min/max over arrays WITHOUT spread — a 7.0.1 AOI run has ~21k KPI points and
// `Math.min(...bigArray)` overflows the call-stack/arg limit (RangeError),
// which would abort the whole load. Reduce is O(n) and size-safe.
const amin = a => a.reduce((m, v) => (v < m ? v : m), Infinity);
const amax = a => a.reduce((m, v) => (v > m ? v : m), -Infinity);

// ---------------------------------------------------------------- bootstrap
async function loadAll() {
  const [summary, points, margins] = await Promise.all([
    fetch("/api/summary").then(r => r.json()),
    fetch("/api/points").then(r => r.json()),
    fetch("/api/defect_margins").then(r => r.json()),
  ]);
  state.summary = summary; state.points = points; state.margins = margins;
  state.iter = summary.best.label !== "—" ? summary.best.label : summary.iterations[0]?.label;
  state.pinned = null; state.hover = null; state.nnSameLabel = null;
  state.space = spaceForIter(state.iter);
  fitView();
  buildHeader(); buildTimeline(); buildFacets(); buildIterInfo(); buildMargins();
  buildRunPicker(); updateSpaceLabel(); resize();
}
loadAll().then(bindControls);

async function buildRunPicker() {
  const runs = await fetch("/api/runs").then(r => r.json());
  const sel = document.getElementById("run-picker");
  sel.innerHTML = runs.map(r =>
    `<option value="${r.name}" ${r.loaded ? "selected" : ""} ${r.standard ? "" : "disabled"}>
      ${r.name}${r.standard ? "" : " (non-standard)"}</option>`).join("");
}

function spaceCoords(p) {
  const c = p.coords?.[state.space];
  return c || null;
}

function fitView() {
  const cs = state.points.map(spaceCoords).filter(Boolean);
  const xs = cs.map(c => c[0]), ys = cs.map(c => c[1]);
  state.bounds = [amin(xs), amax(xs), amin(ys), amax(ys)];
  state.view = { x: 0, y: 0, k: 1 };
}

// SigLIP is the only embedding space: tao-run-deft-aoi mines with SigLIP and
// never emits per-iteration model-space embeddings, so the map does not switch
// spaces when the selected iteration changes.
function spaceForIter(_label) {
  return "siglip";
}

function updateSpaceLabel() {
  document.getElementById("space-label").textContent = "SigLIP (frozen)";
  noteSpace();
}

function noteSpace() {
  const note = document.getElementById("space-note");
  const n = state.points.filter(p => spaceCoords(p)).length;
  note.textContent = `${n}/${state.points.length} points embedded`;
}

// ---------------------------------------------------------------- header/timeline
function buildHeader() {
  const s = state.summary;
  document.getElementById("run-banner").textContent =
    `${s.run_id} · ${s.counts.pool} pool + ${s.counts.kpi} KPI images · target: ${s.kpi_target}`;
  // far_pct is null until an iteration has been evaluated; guard so the whole
  // header still renders instead of throwing on .toFixed of null.
  const far = s.best.far_pct == null ? "n/a" : `${s.best.far_pct.toFixed(2)}%`;
  document.getElementById("kpi-banner").innerHTML =
    `best FAR @ R=100%: <span class="best">${far}</span> (${s.best.label})`;
}

// The timeline tile shows ONLY the row delta, because that is the number that
// reconciles: 255 + 25 = 280. `added_rows` (distinct new images) is smaller
// whenever the loop re-appends an image the set already had, so putting it here
// invites a subtraction that fails — which is exactly how this read before.
// The full reconciliation lives in the iteration panel, where there is room.
function addedLabel(it) {
  return it.appended_rows ? ` (+${it.appended_rows})` : "";
}

// Iteration panel: rows appended, how many were images the model had never seen,
// and how those arrived. Spelled out so the gap is explained rather than implied.
function addedDetail(it) {
  if (!it.appended_rows) return "";
  const by = it.added_by_provenance || {};
  const parts = ["synthetic", "mined", "seed"].filter(k => by[k]).map(k => `${by[k]} ${k}`);
  const split = parts.length ? ` (${parts.join(", ")})` : "";
  const repeats = it.appended_rows - (it.added_rows || 0);
  const repeatNote = repeats > 0
    ? ` · ${repeats} repeat an image already in training`
    : "";
  return `<div><b class="k">added:</b> ${it.appended_rows} rows — `
       + `${it.added_rows} new image${it.added_rows === 1 ? "" : "s"}${split}${repeatNote}</div>`;
}

function buildTimeline() {
  const div = document.getElementById("iters");
  div.innerHTML = "";
  const best = state.summary.best.label;
  let prev = null;
  for (const it of state.summary.iterations) {
    const b = document.createElement("div");
    b.className = "iter-btn" + (it.label === state.iter ? " active" : "");
    const cls = it.label === best ? "best" : (prev !== null && it.far_pct > prev ? "regress" : "");
    b.innerHTML = `<div class="lbl">${it.label}</div>
      <div class="far ${cls}"><span class="unit">FAR</span> ${it.far_pct?.toFixed(1)}%</div>
      <div class="add">${it.train_rows} train rows${addedLabel(it)}</div>`;
    b.onclick = () => {
      state.iter = it.label; state.whatIfThr = null; state.pulseT0 = performance.now();
      buildTimeline(); buildIterInfo(); ensureEdges(); draw();
    };
    div.appendChild(b);
    prev = it.far_pct;
  }
  drawFarChart();
}

function drawFarChart() {
  const c = document.getElementById("farcv"), g = c.getContext("2d");
  c.width = c.parentElement.clientWidth;
  // Only iterations that have actually been evaluated can be plotted. An
  // unevaluated one (hard-stop run, or state written before evaluate) carries
  // far_pct=null, and an unguarded .toFixed() here throws inside loadAll()'s
  // synchronous chain — aborting the whole boot, bindControls included.
  const its = (state.summary.iterations || []).filter(i => i.far_pct != null);
  const W = c.width, H = c.height, pad = 26;
  g.clearRect(0, 0, W, H);
  if (!its.length) return;
  const fars = its.map(i => i.far_pct), mx = Math.max(...fars) + 5, mn = Math.min(...fars) - 5;
  const px = i => its.length === 1 ? W / 2 : pad + i * (W - 2 * pad) / (its.length - 1);
  const py = v => 14 + (H - 46) * (1 - (v - mn) / (mx - mn));
  g.font = "9px sans-serif"; g.textAlign = "left"; g.fillStyle = "#8b94a3";
  g.fillText("FAR @ recall=100% (%) per iteration", 4, 9);
  g.strokeStyle = "#76b900"; g.lineWidth = 2; g.beginPath();
  its.forEach((it, i) => i ? g.lineTo(px(i), py(it.far_pct)) : g.moveTo(px(0), py(it.far_pct)));
  g.stroke();
  g.font = "10px sans-serif"; g.textAlign = "center";
  its.forEach((it, i) => {
    const best = it.label === state.summary.best.label;
    g.fillStyle = best ? "#76b900" : "#8b94a3";
    g.beginPath(); g.arc(px(i), py(it.far_pct), 3.5, 0, 7); g.fill();
    g.fillText(`${it.far_pct.toFixed(1)}%`, px(i), py(it.far_pct) + 14);
    g.fillStyle = best ? "#76b900" : "#5a6270";
    g.fillText(it.label, px(i), H - 3);
  });
}

function buildIterInfo() {
  const it = state.summary.iterations.find(i => i.label === state.iter);
  document.getElementById("iter-detail").innerHTML = `
    <div><b class="k">FAR @ R=100%:</b> ${it.far_pct?.toFixed(2)}% · <b class="k">thr:</b> ${it.threshold?.toFixed(4)}</div>
    <div><b class="k">val_loss:</b> ${it.val_loss ?? "—"} · <b class="k">ckpt:</b> ${it.best_ckpt}</div>
    <div><b class="k">train rows:</b> ${it.train_rows}</div>
    ${addedDetail(it)}
    ${it.mined_by_class && Object.keys(it.mined_by_class).length ? `<div><b class="k">mined by class:</b> ${
      Object.entries(it.mined_by_class).map(([k, v]) => `${k}: ${v}`).join(" · ")}</div>`
      : it.mined_by_class ? `<div><b class="k">mined by class:</b> <span class="note">0 new rows — pool dry around targets (generation candidate)</span></div>` : ""}
    ${it.note ? `<div class="note">${it.note}</div>` : ""}
    <div id="weak-list"></div>`;
  buildWeakList(it);
  buildMargins();
  buildWhatIf(it);
}

// DEFT's own gap-analysis targets for the mining that fed this iteration —
// rendered verbatim from targets_input.parquet, weakest first
async function buildWeakList(it) {
  const host = document.getElementById("weak-list");
  if (!host) return;
  const d = await fetch(`/api/weak_targets/${state.iter}`).then(r => r.json()).catch(() => null);
  if (!d?.targets?.length || document.getElementById("weak-list") !== host) {
    host.innerHTML = "";
    return;
  }
  host.innerHTML = `
    <h3 style="margin:8px 0 3px">DEFT weak samples → mined for ${esc(state.iter)} (${d.targets.length})</h3>
    <div class="weak-chips">${d.targets.map(t =>
      `<span class="weak-chip ${t.label === "PASS" ? "risk" : "wrong"}" ${t.id != null ? `data-id="${t.id}"` : ""}
        title="siamese ${t.siamese_score?.toFixed(4)} · weakness ${t.weakness?.toFixed(4)}">${esc(t.folder)}</span>`).join("")}</div>
    <div class="note">gap-analysis targets, weakest first (hover: siamese / weakness) ·
      amber = PASS, red = NP · click to locate</div>`;
  host.querySelectorAll(".weak-chip[data-id]").forEach(el =>
    el.onclick = () => locatePoint(+el.dataset.id));
}

function buildWhatIf(it) {
  const slider = document.getElementById("thr-slider");
  const scores = state.points.filter(p => p.kind === "kpi").map(p => p.scores?.[state.iter]).filter(s => s !== undefined);
  if (!scores.length || it.threshold == null) { document.getElementById("whatif").hidden = true; return; }
  document.getElementById("whatif").hidden = false;
  slider.min = 0; slider.max = (amax(scores) * 1.1).toFixed(3);
  slider.step = 0.001;
  slider.value = state.whatIfThr ?? it.threshold;
  slider.oninput = () => {
    state.whatIfThr = parseFloat(slider.value);
    updateThrReadout(it); draw();
  };
  document.getElementById("thr-reset").onclick = () => {
    state.whatIfThr = null; slider.value = it.threshold;
    updateThrReadout(it); draw();
  };
  updateThrReadout(it);
}

function updateThrReadout(it) {
  const thr = effThreshold(it);
  const c = kpiConfusion(state.iter, thr);
  const op = kpiConfusion(state.iter, it.threshold);
  const isOp = state.whatIfThr == null || Math.abs(state.whatIfThr - it.threshold) < 1e-9;
  const dFar = c.far - op.far;
  document.getElementById("thr-readout").innerHTML =
    `thr <b>${thr.toFixed(3)}</b> → FAR <b>${c.far.toFixed(1)}%</b> (${c.fp}/${c.nPass}) · ` +
    `recall <b>${c.recall.toFixed(1)}%</b>${c.fn ? ` <span class="delta-bad">(${c.fn} missed!)</span>` : ""}` +
    (isOp ? ` · <span style="color:var(--dim)">operating point</span>`
          : ` · <span class="${dFar <= 0 ? "delta-good" : "delta-bad"}">ΔFAR ${dFar >= 0 ? "+" : ""}${dFar.toFixed(1)}pp</span>`);
  document.getElementById("thr-reset").hidden = isOp;
}

function buildMargins() {
  const rows = state.margins.filter(r => r.iter === state.iter);
  document.getElementById("margin-table").innerHTML = rows.length ? `
    <table><tr><th>KPI defect</th><th>n</th><th>min</th><th>med</th><th>at-risk</th></tr>
    ${rows.map(r => `<tr><td>${dispFacet(r.kpi_defect_type)}</td><td>${r.n}</td>
      <td class="${r.min_margin < 0 ? "neg" : r.min_margin < 0.02 ? "risk" : ""}">${r.min_margin}</td>
      <td>${r.median_margin}</td><td class="${r.at_risk ? "risk" : ""}">${r.at_risk}</td></tr>`).join("")}
    </table><div style="color:var(--dim);font-size:11px;margin-top:4px">margin = siamese_score − threshold; &lt;0 would be a miss</div>` : "<em>no data</em>";
}

// ---------------------------------------------------------------- facets
const FACET_FIELDS = ["label", "provenance", "defect_type", "split", "first_used_iter"];

// display-side vocab normalization — data stays exactly as DEFT emits it
// ('np' in CSVs), but every panel renders the same spelling

const FACET_TITLES = { first_used_iter: "entered training @" };

const NONE = "none";
const ACRONYMS = new Set(["KPI", "SDG", "NP"]);

// Facet values arrive in whatever casing their source used: PASS/NP from the
// CSV sentinel, lowercase provenance and split from the loop, the KPI set's own
// Title_Case defect names. Render them one way. Filtering, colouring and
// counting all key off this same string, so display and data cannot drift.
function dispFacet(v) {
  if (v === NONE) return v;
  return String(v).replace(/_/g, " ").replace(/\S+/g, w =>
    ACRONYMS.has(w.toUpperCase()) ? w.toUpperCase()
      : w[0].toUpperCase() + w.slice(1).toLowerCase());
}

// facet-side view of a point's value; KPI points get their own bucket in
// "entered training @" (they are never trained on) instead of merging into
// the none bucket with the unused pool; label casing normalized; None-string
// and empty defect types merge into one bucket
function facetValue(p, f) {
  if (f === "first_used_iter" && p.kind === "kpi") return dispFacet("kpi");
  const v = p[f];
  return (!v || v === "None") ? NONE : dispFacet(v);
}

function buildFacets() {
  state.facets = {};
  const host = document.getElementById("facets");
  host.innerHTML = "<h3>Filters</h3>";
  for (const f of FACET_FIELDS) {
    const counts = {};
    state.points.forEach(p => { const v = facetValue(p, f); counts[v] = (counts[v] || 0) + 1; });
    const vals = Object.keys(counts).sort();
    if (vals.length < 2) continue;
    state.facets[f] = new Set(vals);
    const g = document.createElement("div"); g.className = "facet-group";
    g.innerHTML = `<b>${FACET_TITLES[f] || f.replaceAll("_", " ")}</b>`;
    for (const v of vals) {
      const l = document.createElement("label");
      l.innerHTML = `<input type="checkbox" checked data-f="${f}" data-v="${v}"> ${v} <span class="cnt" data-f="${f}" data-v="${v}">(${counts[v]})</span>`;
      g.appendChild(l);
    }
    host.appendChild(g);
  }
  host.addEventListener("change", e => {
    const { f, v } = e.target.dataset;
    if (!f) return;
    e.target.checked ? state.facets[f].add(v) : state.facets[f].delete(v);
    updateFacetCounts();
    draw();
  });
  updateFacetCounts();
}

// per-value counts under the current filter, faceted-search style: each
// field's counts honor every OTHER filter (and the search) but not its own,
// so unchecking a value doesn't zero out its siblings
function updateFacetCounts() {
  const spans = {};
  document.querySelectorAll("#facets .cnt").forEach(el => (spans[el.dataset.f] ??= []).push(el));
  for (const [f, els] of Object.entries(spans)) {
    const counts = {};
    state.points.forEach(p => {
      if (!visible(p, f)) return;
      const v = facetValue(p, f);
      counts[v] = (counts[v] || 0) + 1;
    });
    els.forEach(el => { el.textContent = `(${counts[el.dataset.v] || 0})`; });
  }
}

function visible(p, skipFacet = null) {
  if (!spaceCoords(p)) return false;
  for (const f of Object.keys(state.facets)) {
    if (f === skipFacet) continue;
    if (!state.facets[f].has(facetValue(p, f))) return false;
  }
  if (state.search) {
    const hay = `${p.filename || ""} ${p.folder || ""} ${p.dump} ${p.defect_type} ${p.provenance} ${p.split} ${p.label}`.toLowerCase();
    if (!hay.includes(state.search)) return false;
  }
  return true;
}

// ------------------------------------------------------ locate-by-filename
function centerOn(i, zoom = 4) {
  const p = state.points[i];
  const c = spaceCoords(p);
  if (!c) return;
  const [x0, x1, y0, y1] = state.bounds;
  const W = cv.width, H = cv.height, pad = 30 * devicePixelRatio;
  const s = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
  const bx = pad + (c[0] - x0) * s, by = pad + (c[1] - y0) * s;
  state.view.k = Math.max(state.view.k, zoom);
  state.view.x = W / (2 * state.view.k) - bx;
  state.view.y = H / (2 * state.view.k) - by;
}

function locatePoint(i) {
  // clear the text filter so the full map stays visible around the target
  state.search = "";
  const se = document.getElementById("search");
  se.value = "";
  document.getElementById("search-results").hidden = true;
  state.pinned = i;
  state.highlight = new Set([i]);
  state.locateT0 = performance.now();   // drives the locate pulse animation
  centerOn(i);
  showDetails();
  draw();
}

function searchMatches(q) {
  q = q.toLowerCase();
  const out = [];
  for (const p of state.points) {
    const fn = (p.filename || "").toLowerCase(), fo = (p.folder || "").toLowerCase();
    if (fn.includes(q) || fo.includes(q)) {
      out.push(p);
      if (out.length >= 8) break;
    }
  }
  return out;
}

function renderSearchResults(q) {
  const box = document.getElementById("search-results");
  if (!q || q.length < 2) { box.hidden = true; box.innerHTML = ""; return; }
  const matches = searchMatches(q);
  if (!matches.length) { box.hidden = true; box.innerHTML = ""; return; }
  box.innerHTML = matches.map(p =>
    `<div class="sr-item" data-id="${p.id}">
       <span class="fn">${esc(p.filename || p.folder)}</span>
       <span class="meta">${esc(p.kind === "kpi" ? "KPI · " + (p.folder || "") : (p.dump || "") + " · " + p.split)}</span>
     </div>`).join("");
  box.hidden = false;
  box.querySelectorAll(".sr-item").forEach(el =>
    el.onclick = () => { locatePoint(+el.dataset.id); });
  if (matches.length) box.firstElementChild.classList.add("active");
}

// ---------------------------------------------------------------- coloring
function colorMaps() {
  const field = state.colorBy, m = new Map();
  if (field === "status_at_iter") return null;
  let i = 0;
  for (const p of state.points) {
    const v = facetValue(p, field);
    if (!m.has(v)) m.set(v, PALETTE[i++ % PALETTE.length]);
  }
  return m;
}

function pointColor(p, cmap) {
  const it = state.summary.iterations.find(i => i.label === state.iter);
  if (state.colorBy === "status_at_iter") {
    if (p.kind === "kpi") {
      const s = p.scores?.[state.iter];
      if (s === undefined) return "#666";
      const predNP = s > effThreshold(it) + 1e-6; // thresholds rounded to 6dp; == passes
      if (p.label === "PASS" && predNP) return "#e5533c";        // false positive
      if (p.label !== "PASS" && !predNP) return "#ff2dbb";       // miss (none in this run)
      return p.label === "PASS" ? "#76b900" : "#4e9cf5";         // correct
    }
    return "#3a4250";  // pool backdrop
  }
  return cmap.get(facetValue(p, state.colorBy));
}

function inTrainingAtIter(p) {
  if (!p.first_used_iter) return false;
  const order = state.summary.iterations.map(i => i.label);
  return order.indexOf(p.first_used_iter) <= order.indexOf(state.iter);
}

// ---------------------------------------------------------------- canvas
function resize() {
  cv.width = cv.clientWidth * devicePixelRatio;
  cv.height = cv.clientHeight * devicePixelRatio;
  draw();
}
addEventListener("resize", resize);

function world2px(p) {
  const c = spaceCoords(p);
  if (!c) return null;
  const [x0, x1, y0, y1] = state.bounds;
  const W = cv.width, H = cv.height, pad = 30 * devicePixelRatio;
  const sx = (W - 2 * pad) / (x1 - x0), sy = (H - 2 * pad) / (y1 - y0), s = Math.min(sx, sy);
  const bx = pad + (c[0] - x0) * s, by = pad + (c[1] - y0) * s;
  return [(bx + state.view.x) * state.view.k, (by + state.view.y) * state.view.k];
}

function kpiIsWrong(p) {
  const it = state.summary.iterations.find(i => i.label === state.iter);
  const s = p.scores?.[state.iter];
  if (s === undefined || it?.threshold == null) return false;
  const predNP = s > effThreshold(it) + 1e-6;
  return (p.label === "PASS" && predNP) || (p.label !== "PASS" && !predNP);
}

async function ensureEdges() {
  if (!state.showEdges || state.edgesCache[state.iter]) return;
  state.edgesCache[state.iter] = { edges: [] }; // placeholder while fetching
  try {
    state.edgesCache[state.iter] = await fetch(`/api/mining_edges/${state.iter}`).then(r => r.json());
  } catch {
    delete state.edgesCache[state.iter]; // allow retry instead of a stuck empty cache
    return;
  }
  draw();
}

function drawEdges() {
  const payload = state.edgesCache[state.iter];
  if (!payload?.edges?.length) return;
  const pinnedOnly = state.pinned != null;
  for (const e of payload.edges) {
    if (pinnedOnly && e.target !== state.pinned && e.neighbor !== state.pinned) continue;
    const tp = state.points[e.target], np = state.points[e.neighbor];
    if (!visible(tp) || !visible(np)) continue; // edges follow the active filters
    const a = world2px(tp), b = world2px(np);
    if (!a || !b) continue;
    // an edge is "kept" only if THIS retrieval survived: above threshold and
    // the image entered training (a below-threshold edge to an image another
    // target mined above threshold stays gray — it contributed nothing)
    const edgeKept = e.kept && e.above_thr;
    ctx.strokeStyle = edgeKept ? "#76b900" : "#5a6270";
    ctx.globalAlpha = edgeKept ? 0.65 : 0.25;
    ctx.lineWidth = (edgeKept ? 1.4 : 0.8) * devicePixelRatio;
    if (!e.above_thr) ctx.setLineDash([4 * devicePixelRatio, 3 * devicePixelRatio]);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.setLineDash([]);
    if (pinnedOnly) { // cosine readout on the pinned point's own edges
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = edgeKept ? "#76b900" : "#8b94a3";
      ctx.font = `${10 * devicePixelRatio}px sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(e.cosine.toFixed(3), (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 3 * devicePixelRatio);
    }
  }
  ctx.globalAlpha = 1;
}

// Draw a "×N" badge on each screen cell holding 3+ overlapping points, so a
// tight t-SNE stack (e.g. 21 synthetic crops on one spot) is countable without
// zooming. Amber pill, drawn on top of the points.
function drawStackBadges(stacks) {
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.font = `bold ${9 * devicePixelRatio}px sans-serif`;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const s of stacks.values()) {
    if (s.n < 3) continue;
    const label = "×" + s.n;
    const bx = s.x + 4 * devicePixelRatio, by = s.y - 6 * devicePixelRatio;
    const w = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(14,16,20,0.82)";
    ctx.fillRect(bx - 1.5 * devicePixelRatio, by - 6 * devicePixelRatio,
                 w + 3 * devicePixelRatio, 12 * devicePixelRatio);
    ctx.fillStyle = "#e5b83c";
    ctx.fillText(label, bx, by);
  }
  ctx.restore();
}

// Render the fanned-out overlap cluster: legs from the stack centre out to each
// member drawn at its ring position (amber-outlined, in its own color).
function drawSpider(cmap) {
  const legs = spiderLegs();
  if (!legs) { state.spider = null; return; }
  const c = world2px(state.points[state.spider.members[0]]);
  ctx.save();
  ctx.strokeStyle = "rgba(230,184,60,0.55)";
  ctx.lineWidth = 1 * devicePixelRatio;
  for (const L of legs) { ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(L.x, L.y); ctx.stroke(); }
  for (const L of legs) {
    const p = state.points[L.idx];
    const isKpi = p.kind === "kpi";
    const r = (isKpi ? 5 : 4) * devicePixelRatio;
    ctx.fillStyle = pointColor(p, cmap);
    ctx.beginPath();
    if (isKpi) {
      ctx.moveTo(L.x, L.y - r); ctx.lineTo(L.x + r, L.y);
      ctx.lineTo(L.x, L.y + r); ctx.lineTo(L.x - r, L.y); ctx.closePath(); ctx.fill();
    } else { ctx.arc(L.x, L.y, r, 0, 7); ctx.fill(); }
    ctx.strokeStyle = "#e5b83c"; ctx.lineWidth = 1.2 * devicePixelRatio; ctx.stroke();
  }
  ctx.restore();
}

function draw() {
  if (!state.summary) return;
  ctx.fillStyle = "#0e1014"; ctx.fillRect(0, 0, cv.width, cv.height);
  if (state.showEdges) drawEdges();
  const cmap = colorMaps();
  const newest = [];
  // layered draw: pool first, then correct KPI, then failing KPI on top —
  // overlapping diamonds must never hide a failure under a correct one
  const layers = [[], [], []];
  let addedTotal = 0;
  for (const p of state.points) {
    if (p.kind !== "kpi" && p.first_used_iter === state.iter && state.iter !== "baseline") addedTotal++;
    if (!visible(p)) continue;
    if (p.kind !== "kpi") layers[0].push(p);
    else layers[kpiIsWrong(p) ? 2 : 1].push(p);
  }
  // bin visible points into screen cells so co-located (overlapping) points
  // can be badged with a ×N count; cells shrink apart as you zoom in
  const stacks = state.showStacks ? new Map() : null;
  const CELL = 6 * devicePixelRatio;
  for (const layer of layers) for (const p of layer) {
    const xy = world2px(p);
    if (!xy) continue;
    const [x, y] = xy;
    if (stacks) {
      const key = ((x / CELL) | 0) + "," + ((y / CELL) | 0);
      const s = stacks.get(key);
      if (s) s.n++; else stacks.set(key, { n: 1, x, y });
    }
    const inTrain = inTrainingAtIter(p);
    const isKpi = p.kind === "kpi";
    let r = (isKpi ? 5 : 2.6) * devicePixelRatio;
    let alpha = 1;
    if (state.dimUnused && !isKpi && !inTrain) alpha = 0.22;
    if (p.first_used_iter === state.iter && state.iter !== "baseline" && !isKpi) newest.push([x, y, r]);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = pointColor(p, cmap);
    ctx.beginPath();
    if (isKpi) { // diamond for KPI
      ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = "#0e1014"; ctx.lineWidth = 1 * devicePixelRatio; ctx.stroke(); // outline separates stacked diamonds
    } else { ctx.arc(x, y, r, 0, 7); ctx.fill(); }
  }
  ctx.globalAlpha = 1;
  if (stacks) drawStackBadges(stacks);
  if (state.spider) drawSpider(cmap);
  // hover indicator — always matches the tooltip's point (fanned pos if spidered)
  let hoverXY = null;
  if (state.hover != null && state.points[state.hover]) {
    const legs = state.spider ? spiderLegs() : null;
    const leg = legs && legs.find(L => L.idx === state.hover);
    hoverXY = leg ? [leg.x, leg.y] : world2px(state.points[state.hover]);
  }
  if (hoverXY) {
    const [hx, hy] = hoverXY;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.2 * devicePixelRatio;
    ctx.setLineDash([3 * devicePixelRatio, 3 * devicePixelRatio]);
    ctx.beginPath(); ctx.arc(hx, hy, 8 * devicePixelRatio, 0, 7); ctx.stroke();
    ctx.setLineDash([]);
  }
  // glow ring for samples added at the selected iteration (pulses on iteration switch)
  const pulseAge = performance.now() - state.pulseT0;
  const pulsing = pulseAge < 900;
  const pulse = pulsing ? Math.sin((pulseAge / 900) * Math.PI) : 0;
  ctx.strokeStyle = "#e5b83c"; ctx.lineWidth = (1.4 + pulse) * devicePixelRatio;
  ctx.globalAlpha = 0.5 + 0.5 * (1 - pulse);
  for (const [x, y, r] of newest) {
    ctx.beginPath(); ctx.arc(x, y, r + (2.5 + pulse * 6) * devicePixelRatio, 0, 7); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  state.addedInfo = { shown: newest.length, total: addedTotal };
  // run one frame past the pulse window so the ring settles at full opacity
  if (pulseAge < 950) requestAnimationFrame(draw);
  // agent evidence / locate highlight (pulses right after a locate)
  if (state.highlight.size) {
    const locAge = performance.now() - (state.locateT0 || 0);
    const locPulse = locAge < 1600 ? Math.abs(Math.sin(locAge / 200)) : 0;
    ctx.strokeStyle = "#ff9a3c"; ctx.lineWidth = (1.6 + locPulse * 1.5) * devicePixelRatio;
    for (const i of state.highlight) {
      const p = state.points[i]; if (!p) continue;
      const xy = world2px(p); if (!xy) continue;
      const [x, y] = xy;
      ctx.beginPath(); ctx.arc(x, y, (7 + locPulse * 8) * devicePixelRatio, 0, 7); ctx.stroke();
    }
    if (locAge < 1600) requestAnimationFrame(draw);
  }
  // pinned marker — drawn even when facet-filtered out, so locate always lands
  if (state.pinned != null) {
    const p = state.points[state.pinned]; const xy = world2px(p);
    if (xy && !visible(p)) {  // ghost-render a filtered-out pinned point
      const [gx, gy] = xy;
      ctx.globalAlpha = 0.9; ctx.fillStyle = pointColor(p, cmap);
      const r = (p.kind === "kpi" ? 5 : 3) * devicePixelRatio;
      ctx.beginPath(); ctx.arc(gx, gy, r, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
    }
    if (xy) { const [x, y] = xy;
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath(); ctx.arc(x, y, 9 * devicePixelRatio, 0, 7); ctx.stroke(); }
  }
  buildLegend(cmap);
  const n = state.points.filter(visible).length;
  document.getElementById("match-count").textContent = `${n} / ${state.points.length} points shown`;
}

function buildLegend(cmap) {
  const el = document.getElementById("legend");
  let shapeKey = `
    <div><span class="sw dia" style="background:#8b94a3"></span>diamond = KPI (held-out)</div>
    <div><span class="sw" style="background:#8b94a3"></span>circle = pool</div>`;
  const ai = state.addedInfo;
  if (ai?.total) {
    const hidden = ai.shown < ai.total ? ` (${ai.shown}/${ai.total} visible under filters)` : ` (${ai.total})`;
    shapeKey += `
      <div><span class="sw" style="border:1.5px solid #e5b83c;background:none"></span>added this iteration${hidden}</div>`;
  }
  shapeKey += `<div class="legend-sep"></div>`;
  if (state.showEdges) {
    const thr = state.edgesCache[state.iter]?.min_similarity;
    shapeKey += `
      <div><span class="sw edge" style="border-top-color:#76b900"></span>mined via this edge — kept into training</div>
      <div><span class="sw edge" style="border-top-color:#5a6270"></span>above threshold, not kept (e.g. already in training)</div>
      <div><span class="sw edge dashed" style="border-top-color:#5a6270"></span>below cosine threshold${thr != null ? ` (${thr})` : ""} — contributed nothing</div>
      <div class="legend-sep"></div>`;
  }
  if (state.colorBy === "status_at_iter") {
    el.innerHTML = `<h3>Legend</h3>${shapeKey}
      <div><span class="sw dia" style="background:#76b900"></span>KPI PASS — correct</div>
      <div><span class="sw dia" style="background:#e5533c"></span>KPI PASS — false positive</div>
      <div><span class="sw dia" style="background:#4e9cf5"></span>KPI NP — caught</div>
      <div><span class="sw dia" style="background:#ff2dbb"></span>KPI NP — missed</div>
      <div><span class="sw" style="background:#3a4250"></span>pool (dimmed = not in training)</div>`;
    return;
  }
  el.innerHTML = `<h3>Legend</h3>${shapeKey}` + [...cmap].map(([v, c]) =>
    `<div><span class="sw" style="background:${c}"></span>${v}</div>`).join("");
}

// ---------------------------------------------------------------- interaction
function nearest(mx, my) {
  // priority hit-test matching the draw layering: failing KPI > KPI > pool.
  // A KPI diamond within range always beats a pool circle, even a closer one.
  const R = 12 * devicePixelRatio;
  let best = null, bestPri = -1, bd = Infinity;
  state.points.forEach((p, i) => {
    if (!visible(p)) return;
    const xy = world2px(p);
    if (!xy) return;
    const d = Math.hypot(xy[0] - mx, xy[1] - my);
    if (d > R) return;
    const pri = p.kind === "kpi" ? (kpiIsWrong(p) ? 2 : 1) : 0;
    if (pri > bestPri || (pri === bestPri && d < bd)) { bestPri = pri; bd = d; best = i; }
  });
  return best;
}

// --- spiderfy: fan out a stack of overlapping points so each is clickable ---
function clusterAt(mx, my) {
  // visible points whose screen position sits within the overlap radius of (mx,my)
  const R = 9 * devicePixelRatio;
  const hits = [];
  state.points.forEach((p, i) => {
    if (!visible(p)) return;
    const xy = world2px(p);
    if (xy && Math.hypot(xy[0] - mx, xy[1] - my) <= R) hits.push(i);
  });
  return hits;
}

function spiderLegs() {
  // screen positions of the fanned members, on a ring around the stack centre;
  // recomputed each draw from world2px so it tracks pan/zoom
  if (!state.spider) return null;
  const m = state.spider.members;
  const c = world2px(state.points[m[0]]);
  if (!c) return null;
  const R = Math.max(28, 2.6 * m.length) * devicePixelRatio;
  return m.map((idx, i) => {
    const a = (2 * Math.PI * i) / m.length - Math.PI / 2;
    return { idx, x: c[0] + R * Math.cos(a), y: c[1] + R * Math.sin(a) };
  });
}

function spiderHit(mx, my) {
  const legs = spiderLegs();
  if (!legs) return null;
  const R = 10 * devicePixelRatio;
  let best = null, bd = Infinity;
  for (const L of legs) {
    const d = Math.hypot(L.x - mx, L.y - my);
    if (d < R && d < bd) { bd = d; best = L.idx; }
  }
  return best;
}

cv.addEventListener("mousemove", e => {
  const r = cv.getBoundingClientRect();
  // map mouse -> canvas backing-store coords by the TRUE scale, not
  // devicePixelRatio (which is fractional/stale on scaled displays & browser
  // zoom) — keeps the hover ring locked to the "+" cursor.
  const mx = (e.clientX - r.left) * cv.width / r.width;
  const my = (e.clientY - r.top) * cv.height / r.height;
  // when a stack is spidered, hovering a leg previews that specific member
  let i = state.spider ? spiderHit(mx, my) : null;
  if (i == null) i = nearest(mx, my);
  if (i !== state.hover) {
    state.hover = i;
    if (i == null) tooltip.hidden = true;
    else {
      const p = state.points[i];
      tooltip.innerHTML = `<img src="${p.image_url}?thumb=220">
        <div class="cap">${p.folder || p.dump} · ${dispFacet(p.label)}${p.defect_type ? " · " + dispFacet(p.defect_type) : ""}</div>`;
      tooltip.hidden = false;
    }
    draw(); // keep the dashed hover ring in sync with the tooltip
  }
  tooltip.style.left = (e.clientX - r.left + 14) + "px";
  tooltip.style.top = (e.clientY - r.top + 14) + "px";
});
cv.addEventListener("mouseleave", () => { tooltip.hidden = true; state.hover = null; });
cv.addEventListener("click", e => {
  const r = cv.getBoundingClientRect();
  const mx = (e.clientX - r.left) * cv.width / r.width;
  const my = (e.clientY - r.top) * cv.height / r.height;
  // clicking a fanned member pins that specific point; the spider stays open
  if (state.spider) {
    const leg = spiderHit(mx, my);
    if (leg != null) { state.pinned = leg; showDetails(); draw(); return; }
  }
  // clicking an overlapping stack (2+) fans it out for individual access
  const cluster = clusterAt(mx, my);
  if (cluster.length >= 2) {
    state.spider = { members: cluster };
    state.pinned = cluster[0]; showDetails(); draw(); return;
  }
  // otherwise: normal single pin, and collapse any open spider
  state.spider = null;
  state.pinned = nearest(mx, my);
  showDetails(); draw();
});
addEventListener("keydown", e => {
  if (e.key === "Escape" && state.spider) { state.spider = null; draw(); }
});
cv.addEventListener("wheel", e => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  state.view.k = Math.max(0.5, Math.min(12, state.view.k * f));
  draw();
}, { passive: false });
let drag = null;
cv.addEventListener("mousedown", e => drag = [e.clientX, e.clientY]);
addEventListener("mouseup", () => drag = null);
addEventListener("mousemove", e => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  state.view.x += (e.clientX - drag[0]) * (cv.width / r.width) / state.view.k;
  state.view.y += (e.clientY - drag[1]) * (cv.height / r.height) / state.view.k;
  drag = [e.clientX, e.clientY]; draw();
});

function showDetails() {
  const host = document.getElementById("detail-body");
  if (state.pinned == null) { host.innerHTML = "<em>hover / click a point</em>"; return; }
  const p = state.points[state.pinned];
  const it = state.summary.iterations.find(i => i.label === state.iter);
  let scoreRows = "";
  if (p.kind === "kpi" && p.scores) {
    scoreRows = Object.entries(p.scores).map(([k, v]) => {
      const thr = state.summary.iterations.find(i => i.label === k)?.threshold;
      const fp = p.label === "PASS" && v > thr;
      return `<tr><td>score @ ${k}</td><td class="${fp ? "neg" : ""}">${v.toFixed(4)}${fp ? " (FP)" : ""}</td></tr>`;
    }).join("");
  }
  // A component is captured once per lighting condition and a defect may be
  // visible under only one, so offer them all. Single-light runs (every run
  // today) render exactly as before — there is nothing to choose between.
  const lights = state.summary.lights || [];
  const lightBar = lights.length > 1
    ? `<div class="lightbar">${lights.map(l =>
        `<button class="lightbtn${l === state.light ? " on" : ""}" data-light="${l}">${l}</button>`
      ).join("")}</div>`
    : "";
  const lq = state.light && state.light !== lights[0]
    ? `&light=${encodeURIComponent(state.light)}` : "";
  host.innerHTML = `${lightBar}<img src="${p.image_url}?thumb=320${lq}"
      onerror="this.style.visibility='hidden'">
    <table class="meta">
      ${p.folder ? `<tr><td>folder</td><td>${p.folder}</td></tr>` : ""}
      <tr><td>point id</td><td>${p.id}</td></tr>
      ${p.location ? `<tr><td>location</td><td class="loc" title="${esc(p.location)}">${esc(p.location)}</td></tr>` : ""}
      <tr><td>kind</td><td>${p.kind}</td></tr>
      <tr><td>label</td><td>${dispFacet(p.label)}</td></tr>
      ${p.defect_type ? `<tr><td>defect</td><td>${p.kpi_defect_type || p.defect_type}${p.kpi_defect_type && p.defect_type !== p.kpi_defect_type ? ` (≈ ${p.defect_type})` : ""}</td></tr>` : ""}
      <tr><td>provenance</td><td>${p.provenance}</td></tr>
      <tr><td>split</td><td>${p.split}</td></tr>
      ${p.first_used_iter ? `<tr><td>entered training</td><td>${p.first_used_iter}</td></tr>` : ""}
      ${p.dump && p.dump !== "kpi" ? `<tr><td>dump</td><td>${p.dump}</td></tr>` : ""}
      ${p.cam_yaw != null ? `<tr><td>camera</td><td>yaw ${p.cam_yaw.toFixed(0)}° pitch ${p.cam_pitch?.toFixed(0)}°</td></tr>` : ""}
      ${scoreRows}
    </table>
    <div id="nn-box"></div>`;
  // switching lighting re-renders the same component — the point, its score and
  // its position never change, only which capture you are looking at
  host.querySelectorAll(".lightbtn").forEach(b => {
    b.onclick = () => { state.light = b.dataset.light; showDetails(); };
  });
  loadNeighbors(p);
}

async function loadNeighbors(p) {
  const box = document.getElementById("nn-box");
  if (!box) return;
  box.innerHTML = `<div class="note">loading neighbors…</div>`;
  const sl = state.nnSameLabel == null ? "" : `&same_label=${state.nnSameLabel ? 1 : 0}`;
  const d = await fetch(`/api/neighbors/${p.id}?iteration=${state.iter}${sl}`).then(r => r.json());
  if (d.error) { box.innerHTML = `<div class="note">${esc(d.error)}</div>`; return; }
  const rows = d.neighbors.map(nb => {
    const badge = nb.kept ? `<span class="nn-badge kept">kept</span>`
      : nb.retrieved ? `<span class="nn-badge top5">top-${d.topn}</span>` : "";
    return `<tr class="nn-row" data-id="${nb.id}" ${nb.above_thr ? "" : 'style="opacity:.55"'}>
      <td><img src="/api/image/${nb.id}?thumb=44" loading="lazy"></td>
      <td class="nn-fn">${esc(nb.filename)}<br>
        <span class="nn-sub">${esc(nb.provenance)} · ${esc(nb.split)} · ${esc(dispFacet(nb.label))}${nb.defect_type ? "/" + esc(dispFacet(nb.defect_type)) : ""}</span></td>
      <td><b>${nb.cosine.toFixed(3)}</b>${nb.above_thr ? "" : `<span class="nn-sub"> &lt;thr</span>`}<br>${badge}${nb.entered_training ? `<span class="nn-sub">→${esc(nb.entered_training)}</span>` : ""}</td>
    </tr>`;
  }).join("");
  box.innerHTML = `
    <h3 style="margin:10px 0 4px">Nearest neighbors @ ${esc(state.iter)} <span class="nn-sub">(${esc(d.space)})</span></h3>
    ${d.siamese_score != null ? `<div class="note">target siamese score @ ${esc(state.iter)}: <b>${d.siamese_score.toFixed(4)}</b></div>` : ""}
    <table class="nn-table">${rows}</table>
    <label class="note" style="display:block;cursor:pointer">
      <input type="checkbox" id="nn-samelabel" ${d.label_filter_active ? "checked" : ""}>
      restrict to same label${d.label_filter_is_runs
        ? ` (run's own setting: ${d.filter_by_label ? "on" : "off"})`
        : ` — what-if override (run mined with it ${d.filter_by_label ? "on" : "off"})`}</label>
    <div class="note">replay of this run's mining: top-${d.topn} per weak target from the mining pool,
      cosine ≥ ${d.min_similarity}; "kept" = actually entered training</div>`;
  box.querySelectorAll(".nn-row").forEach(el =>
    el.onclick = () => locatePoint(+el.dataset.id));
  document.getElementById("nn-samelabel").onchange = e => {
    state.nnSameLabel = e.target.checked === d.filter_by_label ? null : e.target.checked;
    loadNeighbors(p);
  };
}

function bindControls() {
  document.getElementById("color-by").onchange = e => { state.colorBy = e.target.value; draw(); };
  document.getElementById("dim-unused").onchange = e => { state.dimUnused = e.target.checked; draw(); };
  document.getElementById("show-edges").onchange = e => { state.showEdges = e.target.checked; ensureEdges(); draw(); };
  document.getElementById("show-stacks").onchange = e => { state.showStacks = e.target.checked; draw(); };
  const searchEl = document.getElementById("search");
  searchEl.oninput = e => {
    state.search = e.target.value.trim().toLowerCase();
    renderSearchResults(state.search);
    updateFacetCounts();
    draw();
  };
  searchEl.onkeydown = e => {
    if (e.key === "Enter") {
      const m = searchMatches(state.search);
      if (m.length) locatePoint(m[0].id);
    } else if (e.key === "Escape") {
      document.getElementById("search-results").hidden = true;
    }
  };
  document.getElementById("run-picker").onchange = async e => {
    await fetch(`/api/load/${e.target.value}`, { method: "POST" });
    loadAll();
  };
  document.getElementById("reload-btn").onclick = async () => {
    await fetch("/api/reload", { method: "POST" });
    loadAll();
  };
  bindAgent();
  bindSplitters();
}

// ------------------------------------------------------- resizable panels
function bindSplitters() {
  const root = document.documentElement;
  const saved = JSON.parse(localStorage.getItem("dmc-panels") || "{}");
  if (saved.left) root.style.setProperty("--left-w", saved.left + "px");
  if (saved.right) root.style.setProperty("--right-w", saved.right + "px");
  const setup = (id, side, min, max) => {
    const el = document.getElementById(id);
    el.addEventListener("mousedown", e => {
      e.preventDefault();
      el.classList.add("dragging");
      const move = ev => {
        let w = side === "left" ? ev.clientX : innerWidth - ev.clientX;
        w = Math.max(min, Math.min(max, w));
        root.style.setProperty(`--${side}-w`, w + "px");
        saved[side] = w;
        requestAnimationFrame(resize); // canvas tracks the center column
      };
      const up = () => {
        el.classList.remove("dragging");
        removeEventListener("mousemove", move); removeEventListener("mouseup", up);
        localStorage.setItem("dmc-panels", JSON.stringify(saved));
        resize(); drawFarChart();
      };
      addEventListener("mousemove", move); addEventListener("mouseup", up);
    });
    el.addEventListener("dblclick", () => {   // double-click = reset default
      delete saved[side];
      root.style.setProperty(`--${side}-w`, side === "left" ? "230px" : "330px");
      localStorage.setItem("dmc-panels", JSON.stringify(saved));
      resize(); drawFarChart();
    });
  };
  setup("split-left", "left", 140, 480);
  setup("split-right", "right", 240, 700);
}

// ================================================================ RCA AGENT
function bindAgent() {
  document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === t));
    document.getElementById("tab-analytics").hidden = t.dataset.tab !== "analytics";
    document.getElementById("tab-agent").hidden = t.dataset.tab !== "agent";
  });
  fetch("/api/agent/config").then(r => r.json()).then(c => {
    document.getElementById("agent-model").textContent =
      `· ${c.model || "no model set"}${c.key_set ? "" : " · KEY MISSING"}`;
  });
  const form = document.getElementById("chat-form"), input = document.getElementById("chat-input");
  form.onsubmit = e => { e.preventDefault(); send(input.value); input.value = ""; };
  document.querySelectorAll(".sugg").forEach(b => b.onclick = () => send(b.textContent));
  document.getElementById("chat-reset").onclick = async () => {
    await fetch("/api/agent/reset", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    document.getElementById("chat-log").innerHTML = "";
    state.transcript = []; state.highlight.clear(); draw();
  };
  document.getElementById("clear-highlight").onclick = () => { state.highlight.clear(); draw(); };
  document.getElementById("chat-export").onclick = exportReport;
}

function chatEl(cls, html) {
  const log = document.getElementById("chat-log");
  const d = document.createElement("div");
  d.className = cls; d.innerHTML = html;
  log.appendChild(d); log.scrollTop = log.scrollHeight;
  return d;
}

async function send(text) {
  text = text.trim(); if (!text) return;
  if (state.pinned != null) {
    const p = state.points[state.pinned];
    if (p.folder && !text.includes(p.folder)) text += ` (pinned sample: ${p.folder}, point id ${p.id})`;
  }
  chatEl("msg user", esc(text));
  state.transcript.push({ role: "user", text });
  const resp = await fetch("/api/agent/chat", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text, session: "default" }),
  });
  const reader = resp.body.getReader(), dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 2);
      if (!line.startsWith("data: ")) continue;
      handleEvent(JSON.parse(line.slice(6)));
    }
  }
}

function handleEvent(ev) {
  if (ev.type === "thinking") {
    const full = ev.text.trim();
    // compact by default: a one-sentence preview; click toggles full/collapsed
    const sentence = (full.match(/^[\s\S]*?[.!?](\s|$)/) || [full])[0].trim();
    const preview = sentence.length > 160 ? sentence.slice(0, 160) + "…" : sentence;
    const hasMore = preview.length < full.length;
    const render = (open) =>
      `<span class="th-caret">${hasMore ? (open ? "▾" : "▸") : "·"}</span> ` +
      `<span class="th-label">🧠 thinking</span> ` +
      `<span class="th-body">${esc(open ? full : preview)}</span>`;
    const el = chatEl("msg thinking", render(false));
    if (hasMore) {
      el.style.cursor = "pointer";
      el.title = "click to expand / collapse";
      let open = false;
      el.onclick = () => { open = !open; el.innerHTML = render(open); };
    }
  } else if (ev.type === "text") {
    const { body, verdict } = splitVerdict(ev.text);
    if (body.trim()) {
      const el = chatEl("msg agent", linkifyPoints(mdToHtml(body.trim())));
      bindPointLinks(el);
      state.transcript.push({ role: "agent", text: body.trim() });
    }
    if (verdict) {
      const findings = (verdict.findings || []).map(f => `<li>${linkifyPoints(esc(f))}</li>`).join("");
      const el = chatEl("verdict-card",
        `<b>VERDICT: ${esc(verdict.name || verdict.verdict || "")}</b>` +
        (verdict.confidence ? ` · ${esc(verdict.confidence)} confidence` : "") +
        (verdict.summary ? `<br>${linkifyPoints(esc(verdict.summary))}` : "") +
        (findings ? `<ul>${findings}</ul>` : ""));
      bindPointLinks(el);
      state.transcript.push({ role: "verdict", text: JSON.stringify(verdict) });
    }
  } else if (ev.type === "tool_call") {
    // shown when its result arrives; keep log tight
  } else if (ev.type === "tool_result") {
    const ids = ev.point_ids || [];
    const chip = chatEl("toolchip",
      `⚙ ${esc(ev.name)} — ${esc(ev.summary || "ok")}${ids.length ? ` · <span class="n">${ids.length} pts — click to highlight</span>` : ""}`);
    state.transcript.push({ role: "tool", text: `${ev.name}: ${ev.summary}` });
    if (ids.length) chip.onclick = () => { state.highlight = new Set(ids); draw(); };
  } else if (ev.type === "error") {
    chatEl("msg error", esc(ev.text));
  }
}

function splitVerdict(text) {
  // accept any fenced block whose JSON carries a verdict shape (name/verdict/
  // findings) — models drift on the fence tag (```verdict / ```json / ```),
  // casing, and trailing commas; the 7.0.1 schema uses "name".
  const re = /```[a-z]*\s*([\s\S]*?)```/gi;
  let m;
  while ((m = re.exec(text)) !== null) {
    const v = _parseVerdict(m[1]);
    if (v) return { body: text.replace(m[0], ""), verdict: v };
  }
  return { body: text, verdict: null };
}

function _parseVerdict(raw) {
  // try the raw block, then a lightly-repaired version (strip trailing commas)
  for (const s of [raw.trim(), raw.trim().replace(/,\s*([}\]])/g, "$1")]) {
    try {
      const v = JSON.parse(s);
      if (v && (v.name !== undefined || v.verdict !== undefined || v.findings !== undefined))
        return v;
    } catch { /* keep trying */ }
  }
  return null;
}

// Turn "point 13172", "point_id 13172", "points 12, 34" in agent HTML into
// clickable links that locate the point on the map. Runs on rendered HTML.
function linkifyPoints(html) {
  return html.replace(
    /(point(?:[ _]?id)?s?\b[\s:#]*)((?:\d{1,6})(?:\s*,\s*\d{1,6})*)/gi,
    (_full, pre, nums) => pre + nums.replace(/\d{1,6}/g,
      n => `<a class="pt-link" data-id="${n}" title="show point ${n} on the map"` +
           ` style="color:#4ea1ff;cursor:pointer;text-decoration:underline">${n}</a>`));
}

function bindPointLinks(el) {
  if (!el) return;
  el.querySelectorAll(".pt-link").forEach(a =>
    a.onclick = () => locatePoint(+a.dataset.id));
}

function exportReport() {
  const s = state.summary;
  const lines = [`# RCA Agent Report — ${s.run_id}`, "",
    `Best FAR @ recall=100%: **${s.best.far_pct}%** (${s.best.label})`, "", "---", ""];
  for (const t of state.transcript) {
    if (t.role === "user") lines.push(`## Q: ${t.text}`, "");
    else if (t.role === "tool") lines.push(`> ⚙ ${t.text}`, "");
    else if (t.role === "verdict") lines.push("**" + t.text + "**", "");
    else lines.push(t.text, "");
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `RCA_Agent_Report_${s.run_id}.md`;
  a.click();
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// ---- minimal, safe markdown -> HTML (tables, headers, lists, bold, code) ----
function mdInline(s) {
  return s
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
}
function mdRow(line) {
  return line.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(c => c.trim());
}
function mdToHtml(src) {
  const TH = "border:1px solid #2a2f3a;padding:3px 8px;text-align:left;background:#1b2a08;color:#a6e22e";
  const TD = "border:1px solid #2a2f3a;padding:3px 8px;text-align:left";
  const lines = esc(src).split("\n");   // escape first, then render markdown
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // fenced code block ```lang … ``` -> <pre> (final guard: a verdict that
    // failed to parse must never leak as raw backticks into the transcript)
    if (/^\s*```/.test(line)) {
      const code = []; i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) { code.push(lines[i]); i++; }
      i++;  // consume the closing fence
      out.push(`<pre style="background:#0d1117;border:1px solid #2a2f3a;border-radius:6px;` +
               `padding:8px 10px;overflow:auto;font-size:12px;margin:6px 0">${code.join("\n")}</pre>`);
      continue;
    }
    // table: a row with pipes followed by a |---|--- separator
    if (/\|/.test(line) && i + 1 < lines.length &&
        /-/.test(lines[i + 1]) && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1])) {
      const head = mdRow(line); i += 2;
      const rows = [];
      while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim() !== "") { rows.push(mdRow(lines[i])); i++; }
      let t = `<table style="border-collapse:collapse;margin:6px 0;font-size:12px"><thead><tr>`;
      t += head.map(h => `<th style="${TH}">${mdInline(h)}</th>`).join("") + "</tr></thead><tbody>";
      t += rows.map(r => "<tr>" + r.map(c => `<td style="${TD}">${mdInline(c)}</td>`).join("") + "</tr>").join("");
      out.push(t + "</tbody></table>");
      continue;
    }
    let m;
    if ((m = line.match(/^\s*(#{1,4})\s+(.*)$/))) { const n = Math.min(m[1].length + 2, 6); out.push(`<h${n}>${mdInline(m[2])}</h${n}>`); i++; continue; }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { items.push(`<li>${mdInline(lines[i].replace(/^\s*[-*]\s+/, ""))}</li>`); i++; }
      out.push("<ul style='margin:4px 0 4px 18px'>" + items.join("") + "</ul>");
      continue;
    }
    if (line.trim() === "") { i++; continue; }
    const para = [line]; i++;
    while (i < lines.length && lines[i].trim() !== "" && !/\|/.test(lines[i]) &&
           !/^\s*[-*]\s+/.test(lines[i]) && !/^\s*#{1,4}\s+/.test(lines[i])) { para.push(lines[i]); i++; }
    out.push("<p>" + mdInline(para.join("<br>")) + "</p>");
  }
  return out.join("");
}
