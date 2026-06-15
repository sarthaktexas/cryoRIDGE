/**
 * Thesis Graphical Abstract — Figma plugin
 *
 * Renders a graphical abstract from real pooled cohort data:
 *   (a) BlocRes local resolution vs Q-score
 *   (b) Reliability score vs Q-score
 *   (c) Mean Q by reliability decile (+ optional low-Q AUC mini-bars)
 *
 * Data: embedded in ui.html via scripts/run_graphical_abstract_export.py
 */

figma.showUI(__html__, { width: 320, height: 680 });

// ─── Palette ─────────────────────────────────────────────────────────────────

function hexToRgb(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16) / 255,
    g: parseInt(hex.slice(3, 5), 16) / 255,
    b: parseInt(hex.slice(5, 7), 16) / 255,
  };
}

const PALETTE = {
  red: hexToRgb("#E8303A"),
  cyan: hexToRgb("#30C8E8"),
  green: hexToRgb("#3BBF6A"),
  blue: hexToRgb("#4B6FD4"),
  purple: hexToRgb("#8B84D7"),
  amber: hexToRgb("#F5C518"),
  ink: { r: 0.12, g: 0.12, b: 0.14 },
  muted: { r: 0.42, g: 0.42, b: 0.46 },
  grid: { r: 0.88, g: 0.88, b: 0.9 },
  bg: { r: 1, g: 1, b: 1 },
  panelBg: { r: 0.985, g: 0.985, b: 0.99 },
};

const FONT_CANDIDATES = [
  { family: "Inter", style: "Regular" },
  { family: "Inter", style: "Medium" },
  { family: "Helvetica", style: "Regular" },
];

const MONO_CANDIDATES = [
  { family: "Geist Mono", style: "Regular" },
  { family: "Roboto Mono", style: "Regular" },
  { family: "Inter", style: "Regular" },
];

/** @type {FontName | null} */
let cachedBodyFont = null;
/** @type {FontName | null} */
let cachedMonoFont = null;

// ─── Fonts ───────────────────────────────────────────────────────────────────

async function loadBodyFont() {
  if (cachedBodyFont) return cachedBodyFont;
  for (let i = 0; i < FONT_CANDIDATES.length; i++) {
    try {
      await figma.loadFontAsync(FONT_CANDIDATES[i]);
      cachedBodyFont = FONT_CANDIDATES[i];
      return cachedBodyFont;
    } catch (e) { /* next */ }
  }
  throw new Error("Could not load Inter/Helvetica.");
}

async function loadMonoFont() {
  if (cachedMonoFont) return cachedMonoFont;
  for (let i = 0; i < MONO_CANDIDATES.length; i++) {
    try {
      await figma.loadFontAsync(MONO_CANDIDATES[i]);
      cachedMonoFont = MONO_CANDIDATES[i];
      return cachedMonoFont;
    } catch (e) { /* next */ }
  }
  return loadBodyFont();
}

// ─── Drawing helpers ─────────────────────────────────────────────────────────

function solidFill(color, opacity) {
  return {
    type: "SOLID",
    color: { r: color.r, g: color.g, b: color.b },
    opacity: opacity != null ? opacity : 1,
  };
}

function fmtRho(v) {
  if (!isFinite(v)) return "n/a";
  return (v >= 0 ? "+" : "") + v.toFixed(3);
}

function axisBounds(values, padFrac) {
  padFrac = padFrac != null ? padFrac : 0.06;
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!isFinite(min)) return { min: 0, max: 1 };
  if (min === max) {
    const d = Math.abs(min) * 0.1 || 0.1;
    return { min: min - d, max: max + d };
  }
  const span = max - min;
  return { min: min - span * padFrac, max: max + span * padFrac };
}

/**
 * @param {FrameNode} parent
 */
function addLine(parent, x0, y0, x1, y1, style) {
  style = style || {};
  const vec = figma.createVector();
  vec.vectorPaths = [{
    windingRule: "NONZERO",
    data: "M " + x0 + " " + y0 + " L " + x1 + " " + y1,
  }];
  vec.strokes = [{
    type: "SOLID",
    color: style.color || PALETTE.muted,
    opacity: style.opacity != null ? style.opacity : 1,
  }];
  vec.strokeWeight = style.weight != null ? style.weight : 1;
  if (style.dash) vec.dashPattern = style.dash;
  vec.fills = [];
  parent.appendChild(vec);
  return vec;
}

function addText(parent, text, x, y, font, opts) {
  opts = opts || {};
  const node = figma.createText();
  node.fontName = font;
  node.fontSize = opts.size != null ? opts.size : 11;
  node.characters = text;
  node.fills = [{ type: "SOLID", color: opts.color || PALETTE.ink }];
  node.textAlignHorizontal = opts.align || "LEFT";
  node.x = x;
  node.y = y;
  if (opts.width) {
    node.textAutoResize = "HEIGHT";
    node.resize(opts.width, 20);
  }
  parent.appendChild(node);
  return node;
}

function linearFit(xs, ys) {
  const n = xs.length;
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-9) return { slope: 0, intercept: sy / n };
  const slope = (n * sxy - sx * sy) / denom;
  return { slope: slope, intercept: (sy - slope * sx) / n };
}

/**
 * @param {string} panelLetter
 * @param {*} panelDef
 * @param {FontName} bodyFont
 * @param {FontName} monoFont
 * @param {*} cohortMeta
 */
function buildScatterPanel(panelLetter, panelDef, bodyFont, monoFont, cohortMeta) {
  const W = panelDef.width || 268;
  const H = panelDef.height || 228;
  const padL = 42;
  const padR = 12;
  const padT = 40;
  const padB = 38;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const points = panelDef.points || [];
  const xs = points.map(function (p) { return p.x; });
  const ys = points.map(function (p) { return p.y; });
  const xb = axisBounds(xs);
  const yb = axisBounds(ys);

  const frame = figma.createFrame();
  frame.name = "panel: " + panelDef.title;
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panelLetter, 8, 6, bodyFont, { size: 12, color: PALETTE.muted });
  addText(frame, panelDef.title, 24, 6, bodyFont, { size: 10, color: PALETTE.ink, width: W - 30 });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  for (let g = 1; g < 4; g++) {
    addLine(frame, x0 + (plotW * g) / 4, y0, x0 + (plotW * g) / 4, y1, { color: PALETTE.grid, weight: 0.5 });
    addLine(frame, x0, y0 + (plotH * g) / 4, x1, y0 + (plotH * g) / 4, { color: PALETTE.grid, weight: 0.5 });
  }
  addLine(frame, x0, y1, x1, y1, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0, y0, x0, y1, { color: PALETTE.ink, weight: 1 });

  function dataToPx(x, y) {
    const tx = (x - xb.min) / (xb.max - xb.min);
    const ty = (y - yb.min) / (yb.max - yb.min);
    return { px: x0 + tx * plotW, py: y1 - ty * plotH };
  }

  const fit = linearFit(xs, ys);
  const pLine0 = dataToPx(xb.min, fit.slope * xb.min + fit.intercept);
  const pLine1 = dataToPx(xb.max, fit.slope * xb.max + fit.intercept);
  const trendColor = panelDef.trendColor || PALETTE.cyan;
  addLine(frame, pLine0.px, pLine0.py, pLine1.px, pLine1.py, {
    color: trendColor,
    weight: 1.5,
    opacity: 0.9,
  });

  const ptR = points.length > 800 ? 2.2 : 2.8;
  const ptColor = panelDef.pointColor || PALETTE.red;
  for (let i = 0; i < points.length; i++) {
    const px = dataToPx(points[i].x, points[i].y);
    const dot = figma.createEllipse();
    dot.resize(ptR * 2, ptR * 2);
    dot.x = px.px - ptR;
    dot.y = px.py - ptR;
    dot.fills = [solidFill(ptColor, 0.55)];
    dot.strokes = [];
    frame.appendChild(dot);
  }

  addText(frame, panelDef.x_label, x0, y1 + 8, bodyFont, {
    size: 8,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yLabel = addText(frame, panelDef.y_label, 2, y0 + plotH * 0.28, bodyFont, {
    size: 8,
    color: PALETTE.muted,
  });
  yLabel.rotation = -90;

  const rho = panelDef.spearman_rho;
  addText(
    frame,
    "ρ = " + fmtRho(rho),
    x0,
    y0 - 8,
    monoFont,
    { size: 9, color: trendColor }
  );
  addText(
    frame,
    "n=" + cohortMeta.n_residues_pooled.toLocaleString() + " Cα",
    x1 - 72,
    y0 - 8,
    monoFont,
    { size: 8, color: PALETTE.muted }
  );

  if (cohortMeta.scope && cohortMeta.scope.emdb_id) {
    addText(
      frame,
      "EMD-" + cohortMeta.scope.emdb_id,
      x0,
      H - 12,
      bodyFont,
      { size: 7.5, color: PALETTE.muted }
    );
  }

  return frame;
}

/**
 * Panel (c): reliability decile → mean Q (real aggregated cohort bins).
 */
function buildCalibrationPanel(panelDef, bodyFont, monoFont, cohortMeta) {
  const W = 200;
  const H = 228;
  const padL = 36;
  const padR = 10;
  const padT = 40;
  const padB = 36;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const bins = panelDef.bins || [];

  const frame = figma.createFrame();
  frame.name = "panel: " + panelDef.title;
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, "c", 8, 6, bodyFont, { size: 12, color: PALETTE.muted });
  addText(frame, panelDef.title, 24, 6, bodyFont, { size: 9, color: PALETTE.ink, width: W - 28 });

  const yVals = bins.map(function (b) { return b.mean_q; });
  const yb = axisBounds(yVals, 0.08);
  const x0 = padL;
  const y1 = padT + plotH;

  addLine(frame, x0, padT, x0, y1, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0, y1, x0 + plotW, y1, { color: PALETTE.ink, weight: 1 });

  const nBins = bins.length;
  const gap = 3;
  const barW = (plotW - gap * (nBins - 1)) / Math.max(nBins, 1);

  for (let i = 0; i < nBins; i++) {
    const b = bins[i];
    const t = (b.mean_q - yb.min) / (yb.max - yb.min);
    const barH = Math.max(2, t * plotH);
    const bx = x0 + i * (barW + gap);
    const by = y1 - barH;
    const rect = figma.createRectangle();
    rect.resize(barW, barH);
    rect.x = bx;
    rect.y = by;
    const frac = i / Math.max(nBins - 1, 1);
    rect.fills = [solidFill(
      {
        r: PALETTE.red.r + (PALETTE.green.r - PALETTE.red.r) * frac,
        g: PALETTE.red.g + (PALETTE.green.g - PALETTE.red.g) * frac,
        b: PALETTE.red.b + (PALETTE.green.b - PALETTE.red.b) * frac,
      },
      0.85
    )];
    rect.strokes = [];
    rect.cornerRadius = 1;
    frame.appendChild(rect);

    addText(frame, b.label, bx + barW / 2 - 3, y1 + 4, bodyFont, {
      size: 7,
      color: PALETTE.muted,
    });
    if (i === 0 || i === nBins - 1) {
      addText(frame, b.mean_q.toFixed(2), bx, by - 10, monoFont, {
        size: 7,
        color: PALETTE.ink,
      });
    }
  }

  addText(frame, panelDef.x_label, x0, y1 + 18, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yl = addText(frame, panelDef.y_label, 0, padT + plotH * 0.3, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
  });
  yl.rotation = -90;

  addText(
    frame,
    "Q < " + cohortMeta.q_threshold.toFixed(2) + " threshold",
    x0,
    H - 12,
    bodyFont,
    { size: 7, color: PALETTE.muted }
  );

  return frame;
}

/**
 * Mini panel: median-map AUC for low-Q detection (real cohort numbers).
 */
function buildAucMiniPanel(panelDef, bodyFont, monoFont) {
  const bars = (panelDef.bars || []).filter(function (b) {
    return isFinite(b.auc);
  });
  const frame = figma.createFrame();
  frame.name = "panel: low-Q AUC";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;

  const W = 200;
  const H = 88 + bars.length * 22;
  frame.resize(W, H);

  addText(frame, panelDef.title, 10, 8, bodyFont, {
    size: 8.5,
    color: PALETTE.ink,
    width: W - 16,
  });

  const barLeft = 10;
  const barMaxW = W - 52;
  let y = 30;

  for (let i = 0; i < bars.length; i++) {
    const b = bars[i];
    const color = b.id.indexOf("reliability") >= 0 ? PALETTE.green
      : b.id.indexOf("locres") >= 0 ? PALETTE.purple
      : b.id.indexOf("omit") >= 0 ? PALETTE.amber
      : PALETTE.blue;
    const label = b.label.length > 22 ? b.label.slice(0, 20) + "…" : b.label;
    addText(frame, label, barLeft, y, bodyFont, { size: 7, color: PALETTE.muted, width: barMaxW });

    const track = figma.createRectangle();
    track.resize(barMaxW, 8);
    track.x = barLeft;
    track.y = y + 11;
    track.fills = [solidFill(PALETTE.grid)];
    track.strokes = [];
    track.cornerRadius = 2;
    frame.appendChild(track);

    const fill = figma.createRectangle();
    fill.resize(Math.max(4, barMaxW * b.auc), 8);
    fill.x = barLeft;
    fill.y = y + 11;
    fill.fills = [solidFill(color, 0.9)];
    fill.strokes = [];
    fill.cornerRadius = 2;
    frame.appendChild(fill);

    addText(frame, b.auc.toFixed(2), barLeft + barMaxW + 4, y + 9, monoFont, {
      size: 8,
      color: PALETTE.ink,
    });
    y += 28;
  }

  return frame;
}

function buildPipelineSidebar(bodyFont) {
  const frame = figma.createFrame();
  frame.name = "pipeline sidebar";
  frame.layoutMode = "NONE";
  frame.fills = [];
  const W = 136;
  frame.resize(W, 260);

  const steps = [
    { label: "Half-maps", color: PALETTE.blue },
    { label: "Constraint V", color: PALETTE.cyan },
    { label: "Reliability", color: PALETTE.green },
    { label: "Q-score", color: PALETTE.red },
  ];

  let y = 0;
  for (let i = 0; i < steps.length; i++) {
    const box = figma.createFrame();
    box.resize(W - 4, 34);
    box.x = 0;
    box.y = y;
    box.fills = [solidFill(steps[i].color, 0.12)];
    box.strokes = [{ type: "SOLID", color: steps[i].color, opacity: 0.5 }];
    box.strokeWeight = 1;
    box.cornerRadius = 5;
    frame.appendChild(box);
    addText(box, steps[i].label, 8, 10, bodyFont, { size: 8.5, color: PALETTE.ink });
    if (i < steps.length - 1) {
      addLine(frame, W / 2 - 2, y + 34, W / 2 - 2, y + 42, { color: PALETTE.muted, weight: 1 });
    }
    y += 42;
  }
  return frame;
}

function findClearOrigin() {
  let maxX = 0;
  for (let i = 0; i < figma.currentPage.children.length; i++) {
    const node = figma.currentPage.children[i];
    if ("x" in node && "width" in node) {
      maxX = Math.max(maxX, node.x + node.width);
    }
  }
  return { x: maxX > 0 ? maxX + 64 : 80, y: 80 };
}

function enrichPanelDef(key, panels, stats) {
  const copy = JSON.parse(JSON.stringify(panels[key]));
  if (key === "locres_q") {
    copy.pointColor = PALETTE.purple;
    copy.trendColor = PALETTE.cyan;
  } else if (key === "reliability_q") {
    copy.pointColor = PALETTE.red;
    copy.trendColor = PALETTE.green;
  }
  return copy;
}

/**
 * @param {*} opts
 * @param {FontName} bodyFont
 * @param {FontName} monoFont
 */
async function buildGraphicalAbstract(opts, bodyFont, monoFont) {
  const data = opts.cohortData;
  if (!data || !data.panels) {
    throw new Error("Missing cohort data. Run scripts/run_graphical_abstract_export.py");
  }

  const stats = data.stats;
  const panels = data.panels;
  const layout = opts.layout || "horizontal";
  const includeSidebar = opts.includeSidebar !== false && layout !== "panels_only";
  const includeTakeaway = opts.includeTakeaway !== false;

  const panelA = buildScatterPanel(
    "a",
    enrichPanelDef("locres_q", panels, stats),
    bodyFont,
    monoFont,
    data
  );
  const panelB = buildScatterPanel(
    "b",
    enrichPanelDef("reliability_q", panels, stats),
    bodyFont,
    monoFont,
    data
  );
  const panelC = buildCalibrationPanel(panels.calibration, bodyFont, monoFont, data);
  const panelAuc = buildAucMiniPanel(panels.predictor_auc, bodyFont, monoFont);

  const wrapper = figma.createFrame();
  wrapper.name = "graphical abstract (EMD-" + (data.scope && data.scope.emdb_id ? data.scope.emdb_id : data.n_maps) + ")";
  wrapper.layoutMode = "NONE";
  wrapper.clipsContent = false;
  wrapper.fills = [solidFill(PALETTE.bg)];
  wrapper.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 10;

  const margin = 20;
  let cursorY = margin;

  const titleNode = addText(wrapper, opts.title || "Graphical abstract", margin, cursorY, bodyFont, {
    size: 15,
    color: PALETTE.ink,
    width: 900,
  });
  cursorY += titleNode.height + 4;

  const subText = opts.subtitle || (
    (data.scope && data.scope.display_name
      ? "EMD-" + data.scope.emdb_id + " · " + data.scope.display_name
      : "Exemplar map") +
    " (" + (data.scope && data.scope.regime_label ? data.scope.regime_label : "2.5–4 Å") + "): " +
    "ρ(Q, reliability)=" + fmtRho(stats.spearman_q_vs_reliability)
  );
  const subNode = addText(wrapper, subText, margin, cursorY, bodyFont, {
    size: 9.5,
    color: PALETTE.muted,
    width: 900,
  });
  cursorY += subNode.height + 14;

  const contentY = cursorY;
  let contentX = margin;
  let contentW = 0;
  let contentH = 0;
  const gap = 12;

  if (includeSidebar) {
    const sidebar = buildPipelineSidebar(bodyFont);
    sidebar.x = contentX;
    sidebar.y = contentY;
    wrapper.appendChild(sidebar);
    contentX += sidebar.width + 14;
    contentH = Math.max(contentH, sidebar.height);
  }

  if (layout === "two_plus_mini") {
    panelA.x = contentX;
    panelA.y = contentY;
    wrapper.appendChild(panelA);
    panelB.x = contentX + panelA.width + gap;
    panelB.y = contentY;
    wrapper.appendChild(panelB);
    panelC.x = contentX + panelA.width + gap + panelB.width + gap;
    panelC.y = contentY;
    wrapper.appendChild(panelC);
    panelAuc.x = panelC.x;
    panelAuc.y = contentY + panelC.height + 8;
    wrapper.appendChild(panelAuc);
    contentW = panelA.width + gap + panelB.width + gap + panelC.width;
    contentH = Math.max(contentH, panelC.height + 8 + panelAuc.height);
  } else {
    panelA.x = contentX;
    panelA.y = contentY;
    wrapper.appendChild(panelA);
    panelB.x = contentX + panelA.width + gap;
    panelB.y = contentY;
    wrapper.appendChild(panelB);
    panelC.x = contentX + panelA.width + gap + panelB.width + gap;
    panelC.y = contentY;
    wrapper.appendChild(panelC);
    panelAuc.x = panelC.x;
    panelAuc.y = contentY + panelC.height + 6;
    wrapper.appendChild(panelAuc);
    contentW = panelA.width + gap + panelB.width + gap + panelC.width;
    contentH = Math.max(contentH, panelC.height + 6 + panelAuc.height);
  }

  cursorY = contentY + contentH + 14;

  if (includeTakeaway) {
    const takeaway = figma.createFrame();
    takeaway.fills = [solidFill(PALETTE.green, 0.08)];
    takeaway.strokes = [{ type: "SOLID", color: PALETTE.green, opacity: 0.35 }];
    takeaway.strokeWeight = 1;
    takeaway.cornerRadius = 6;
    const tw = contentX - margin + contentW;
    takeaway.resize(tw, 48);
    takeaway.x = margin;
    takeaway.y = cursorY;
    addText(
      takeaway,
      (data.scope && data.scope.display_name
        ? "EMD-" + data.scope.emdb_id + " (" + data.scope.global_resolution_a + " Å, " + data.scope.regime_label + "): "
        : "") +
        data.n_residues_pooled.toLocaleString() + " Cα · " +
        "ρ(Q, reliability)=" + fmtRho(stats.spearman_q_vs_reliability) +
        " · reliability AUC=" + stats.reliability_median_auc.toFixed(2) +
        " vs BlocRes AUC=" + stats.locres_median_auc.toFixed(2) +
        " (Q < " + data.q_threshold.toFixed(2) + ").",
      10,
      8,
      bodyFont,
      { size: 8.5, color: PALETTE.ink, width: tw - 20 }
    );
    wrapper.appendChild(takeaway);
    cursorY += takeaway.height + margin;
  } else {
    cursorY += margin;
  }

  const totalW = contentX - margin + contentW + margin;
  wrapper.resize(Math.max(totalW, 720), cursorY);
  const origin = findClearOrigin();
  wrapper.x = origin.x;
  wrapper.y = origin.y;
  figma.currentPage.appendChild(wrapper);
  return wrapper;
}

function buildSinglePanel(panelId, opts, bodyFont, monoFont) {
  const data = opts.cohortData;
  const panels = data.panels;
  const stats = data.stats;
  let panel;

  if (panelId === "locres_q" || panelId === "reliability_q") {
    panel = buildScatterPanel(
      panelId === "locres_q" ? "a" : "b",
      enrichPanelDef(panelId, panels, stats),
      bodyFont,
      monoFont,
      data
    );
  } else if (panelId === "calibration") {
    panel = buildCalibrationPanel(panels.calibration, bodyFont, monoFont, data);
  } else if (panelId === "predictor_auc") {
    panel = buildAucMiniPanel(panels.predictor_auc, bodyFont, monoFont);
  } else {
    throw new Error("Unknown panel: " + panelId);
  }

  const origin = findClearOrigin();
  panel.x = origin.x;
  panel.y = origin.y;
  figma.currentPage.appendChild(panel);
  return panel;
}

figma.ui.onmessage = async function (msg) {
  try {
    const bodyFont = await loadBodyFont();
    const monoFont = await loadMonoFont();

    if (msg.type === "generateSinglePanel") {
      const panel = buildSinglePanel(msg.panel, msg, bodyFont, monoFont);
      figma.currentPage.selection = [panel];
      figma.viewport.scrollAndZoomIntoView([panel]);
      figma.ui.postMessage({ type: "result", text: "Created \"" + panel.name + "\"." });
      return;
    }

    if (msg.type !== "generateAbstract") return;

    const wrapper = await buildGraphicalAbstract(msg, bodyFont, monoFont);
    figma.currentPage.selection = [wrapper];
    figma.viewport.scrollAndZoomIntoView([wrapper]);
    figma.ui.postMessage({
      type: "result",
      text: "Generated abstract for EMD-" + (msg.cohortData.scope && msg.cohortData.scope.emdb_id
        ? msg.cohortData.scope.emdb_id
        : "?") + ".",
    });
  } catch (err) {
    figma.ui.postMessage({
      type: "error",
      text: err instanceof Error ? err.message : String(err),
    });
  }
};
