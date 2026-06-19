/**
 * Thesis Q vs V Cohort — Figma plugin
 *
 * Recreates the cohort summary figure from qscore_vs_V_cohort.png:
 *   (a) horizontal per-structure ρ bar chart (color = resolution)
 *   (b) ρ vs global resolution scatter + trend
 */

figma.showUI(__html__, { width: 300, height: 480 });

function hexToRgb(hex) {
  return {
    r: parseInt(hex.slice(1, 3), 16) / 255,
    g: parseInt(hex.slice(3, 5), 16) / 255,
    b: parseInt(hex.slice(5, 7), 16) / 255,
  };
}

const PALETTE = {
  red: hexToRgb("#E8303A"),
  blue: hexToRgb("#4B6FD4"),
  magenta: hexToRgb("#BA3EC3"),
  ink: { r: 0.12, g: 0.12, b: 0.14 },
  muted: { r: 0.42, g: 0.42, b: 0.46 },
  grid: { r: 0.88, g: 0.88, b: 0.9 },
  panelBg: { r: 1, g: 1, b: 1 },
  grayBar: { r: 0.6, g: 0.6, b: 0.6 },
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

function solidFill(color, opacity) {
  return {
    type: "SOLID",
    color: { r: color.r, g: color.g, b: color.b },
    opacity: opacity != null ? opacity : 1,
  };
}

function fmtRho(v) {
  if (!isFinite(v)) return "n/a";
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}

function axisBounds(values, padFrac) {
  padFrac = padFrac != null ? padFrac : 0.08;
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

/** Nice tick positions between min and max (matplotlib-style step picking). */
function pickTicks(min, max, targetCount) {
  targetCount = targetCount != null ? targetCount : 5;
  if (!isFinite(min) || !isFinite(max)) return [];
  if (min === max) return [min];
  const span = max - min;
  const rawStep = span / Math.max(targetCount - 1, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const norm = rawStep / mag;
  let step;
  if (norm <= 1) step = mag;
  else if (norm <= 2) step = 2 * mag;
  else if (norm <= 5) step = 5 * mag;
  else step = 10 * mag;
  let t0 = Math.ceil(min / step - 1e-9) * step;
  const ticks = [];
  for (let t = t0; t <= max + step * 0.001; t += step) {
    ticks.push(Math.round(t * 1e4) / 1e4);
  }
  if (min < 0 && max > 0 && ticks.indexOf(0) < 0) {
    ticks.push(0);
    ticks.sort(function (a, b) { return a - b; });
  }
  if (ticks.length === 0) return [min, max];
  return ticks;
}

function formatTick(v, kind) {
  if (Math.abs(v) < 1e-10) return "0";
  if (kind === "rho") {
    return v.toFixed(2);
  }
  if (kind === "resolution") {
    return v.toFixed(1);
  }
  if (Math.abs(v) >= 10) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

/**
 * Draw x-axis tick marks + labels below the plot baseline.
 */
function drawXAxisTicks(frame, x0, yBase, plotW, bounds, ticks, monoFont, kind) {
  const tickLen = 4;
  for (let i = 0; i < ticks.length; i++) {
    const t = ticks[i];
    const tx = (t - bounds.min) / (bounds.max - bounds.min);
    const px = x0 + tx * plotW;
    addLine(frame, px, yBase, px, yBase + tickLen, { color: PALETTE.ink, weight: 0.6 });
    const label = formatTick(t, kind);
    addText(frame, label, px - 14, yBase + tickLen + 1, monoFont, {
      size: 6.5,
      color: PALETTE.muted,
      width: 28,
      align: "CENTER",
    });
  }
}

/**
 * Draw y-axis tick marks + labels left of the plot.
 */
function drawYAxisTicks(frame, xBase, y0, plotH, bounds, ticks, monoFont, kind) {
  const tickLen = 4;
  const y1 = y0 + plotH;
  for (let i = 0; i < ticks.length; i++) {
    const t = ticks[i];
    const ty = (t - bounds.min) / (bounds.max - bounds.min);
    const py = y1 - ty * plotH;
    addLine(frame, xBase - tickLen, py, xBase, py, { color: PALETTE.ink, weight: 0.6 });
    const label = formatTick(t, kind);
    addText(frame, label, xBase - tickLen - 22, py - 4, monoFont, {
      size: 6.5,
      color: PALETTE.muted,
      width: 20,
      align: "RIGHT",
    });
  }
}

function viridisRgb(t) {
  const stops = [
    [0.267, 0.004, 0.329],
    [0.282, 0.14, 0.458],
    [0.253, 0.265, 0.529],
    [0.206, 0.371, 0.553],
    [0.163, 0.471, 0.558],
    [0.127, 0.566, 0.55],
    [0.134, 0.658, 0.517],
    [0.246, 0.736, 0.433],
    [0.477, 0.821, 0.318],
    [0.741, 0.873, 0.15],
    [0.993, 0.906, 0.144],
  ];
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[Math.min(i, stops.length - 1)];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  return {
    r: a[0] + (b[0] - a[0]) * f,
    g: a[1] + (b[1] - a[1]) * f,
    b: a[2] + (b[2] - a[2]) * f,
  };
}

function resolutionColor(res, vmin, vmax) {
  if (!isFinite(res) || !isFinite(vmin) || !isFinite(vmax) || vmax === vmin) {
    return PALETTE.grayBar;
  }
  return viridisRgb((res - vmin) / (vmax - vmin));
}

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

function addColorbar(parent, x, y, w, h, vmin, vmax, label, bodyFont) {
  const nSteps = 20;
  const stepH = h / nSteps;
  for (let i = 0; i < nSteps; i++) {
    const t = 1 - i / Math.max(nSteps - 1, 1);
    const rect = figma.createRectangle();
    rect.resize(w, stepH + 0.5);
    rect.x = x;
    rect.y = y + i * stepH;
    rect.fills = [solidFill(viridisRgb(t))];
    rect.strokes = [];
    parent.appendChild(rect);
  }
  const border = figma.createRectangle();
  border.resize(w, h);
  border.x = x;
  border.y = y;
  border.fills = [];
  border.strokes = [{ type: "SOLID", color: PALETTE.muted, opacity: 0.5 }];
  border.strokeWeight = 0.5;
  parent.appendChild(border);
  addText(parent, vmax.toFixed(1), x + w + 3, y - 1, bodyFont, { size: 6.5, color: PALETTE.muted });
  addText(parent, vmin.toFixed(1), x + w + 3, y + h - 7, bodyFont, { size: 6.5, color: PALETTE.muted });
  const lbl = addText(parent, label, x + w + 12, y + h * 0.35, bodyFont, { size: 7, color: PALETTE.muted });
  lbl.rotation = -90;
}

/**
 * Panel A: horizontal bar chart (ρ per structure, color = resolution).
 */
function buildCohortPanelA(panel, bodyFont, monoFont) {
  const bars = panel.bars || [];
  const n = bars.length;
  const labelW = 118;
  const padT = 38;
  const padB = 40;
  const padR = 48;
  const barH = 9;
  const barGap = 2;
  const plotH = n * (barH + barGap) - barGap;
  const W = 420;
  const H = padT + plotH + padB;
  const plotW = W - labelW - padR - 18;
  const padL = labelW + 8;

  const rhos = bars.map(function (b) { return b.rho; });
  const xb = axisBounds(rhos, 0.05);
  if (xb.min > 0) xb.min = 0;
  if (xb.max < 0) xb.max = 0;

  const vmin = panel.resolution_min_a != null ? panel.resolution_min_a : 0;
  const vmax = panel.resolution_max_a != null ? panel.resolution_max_a : 1;
  const medianRho = panel.median_rho;

  const frame = figma.createFrame();
  frame.name = "panel a: cohort ρ(Q,V) ranking";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.letter || "a", 8, 6, bodyFont, { size: 12, color: PALETTE.muted });
  addText(frame, panel.title || "", 24, 6, bodyFont, { size: 9.5, color: PALETTE.ink, width: W - 30 });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  function rhoToX(rho) {
    return x0 + ((rho - xb.min) / (xb.max - xb.min)) * plotW;
  }

  if (xb.min <= 0 && xb.max >= 0) {
    const zx = rhoToX(0);
    addLine(frame, zx, y0 - 2, zx, y1 + 2, { color: PALETTE.muted, weight: 0.6, opacity: 0.7 });
  }
  if (isFinite(medianRho) && medianRho >= xb.min && medianRho <= xb.max) {
    const mx = rhoToX(medianRho);
    addLine(frame, mx, y0 - 2, mx, y1 + 2, {
      color: PALETTE.blue,
      weight: 1,
      dash: [3, 2],
    });
  }

  addLine(frame, x0, y1 + 4, x1, y1 + 4, { color: PALETTE.ink, weight: 1 });
  drawXAxisTicks(frame, x0, y1 + 4, plotW, xb, pickTicks(xb.min, xb.max, 5), monoFont, "rho");

  for (let i = 0; i < n; i++) {
    const b = bars[i];
    const by = y0 + i * (barH + barGap);
    const xStart = rhoToX(Math.min(0, b.rho));
    const xEnd = rhoToX(Math.max(0, b.rho));
    const bw = Math.max(1, Math.abs(xEnd - xStart));
    const bx = Math.min(xStart, xEnd);

    const rect = figma.createRectangle();
    rect.resize(bw, barH);
    rect.x = bx;
    rect.y = by;
    rect.fills = [solidFill(resolutionColor(b.resolution_a, vmin, vmax), 0.92)];
    rect.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
    rect.strokeWeight = 0.4;
    rect.cornerRadius = 1;
    frame.appendChild(rect);

    const lbl = b.label || b.emdb_id;
    const shortLbl = lbl.length > 22 ? lbl.slice(0, 20) + "…" : lbl;
    addText(frame, shortLbl, 4, by + 1, bodyFont, {
      size: 6.5,
      color: PALETTE.ink,
      width: labelW - 6,
      align: "RIGHT",
    });
  }

  addText(frame, panel.x_label || "", x0, y1 + 22, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });

  addColorbar(
    frame,
    x1 + 6,
    y0,
    8,
    Math.min(plotH, 120),
    vmin,
    vmax,
    panel.color_label || "Global resolution (Å)",
    bodyFont
  );

  return frame;
}

/**
 * Panel B: ρ vs global resolution scatter + linear trend.
 */
function buildCohortPanelB(panel, bodyFont, monoFont) {
  const points = panel.points || [];
  const W = 300;
  const H = Math.max(280, 120 + points.length * 2);
  const padL = 52;
  const padR = 14;
  const padT = 38;
  const padB = 44;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xs = points.map(function (p) { return p.resolution_a; });
  const ys = points.map(function (p) { return p.rho; });
  const xb = axisBounds(xs);
  const yb = axisBounds(ys);

  const frame = figma.createFrame();
  frame.name = "panel b: ρ(Q,V) vs resolution";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.letter || "b", 8, 6, bodyFont, { size: 12, color: PALETTE.muted });
  addText(frame, panel.title || "", 24, 6, bodyFont, { size: 9, color: PALETTE.ink, width: W - 28 });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  function dataToPx(x, y) {
    const tx = (x - xb.min) / (xb.max - xb.min);
    const ty = (y - yb.min) / (yb.max - yb.min);
    return { px: x0 + tx * plotW, py: y1 - ty * plotH };
  }

  for (let g = 1; g < 4; g++) {
    addLine(frame, x0 + (plotW * g) / 4, y0, x0 + (plotW * g) / 4, y1, { color: PALETTE.grid, weight: 0.5 });
    addLine(frame, x0, y0 + (plotH * g) / 4, x1, y0 + (plotH * g) / 4, { color: PALETTE.grid, weight: 0.5 });
  }
  addLine(frame, x0, y1, x1, y1, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0, y0, x0, y1, { color: PALETTE.ink, weight: 1 });

  const xTicksB = pickTicks(xb.min, xb.max, 5);
  const yTicksB = pickTicks(yb.min, yb.max, 5);
  drawXAxisTicks(frame, x0, y1, plotW, xb, xTicksB, monoFont, "resolution");
  drawYAxisTicks(frame, x0, y0, plotH, yb, yTicksB, monoFont, "rho");

  if (yb.min <= 0 && yb.max >= 0) {
    const zy = dataToPx(xb.min, 0).py;
    addLine(frame, x0, zy, x1, zy, { color: PALETTE.muted, weight: 0.6, opacity: 0.7 });
  }

  const trend = panel.trend_line || [];
  if (trend.length >= 2) {
    const p0 = dataToPx(trend[0].x, trend[0].y);
    const p1 = dataToPx(trend[trend.length - 1].x, trend[trend.length - 1].y);
    addLine(frame, p0.px, p0.py, p1.px, p1.py, { color: PALETTE.blue, weight: 1 });
  }

  const ptR = 5;
  for (let i = 0; i < points.length; i++) {
    const px = dataToPx(points[i].resolution_a, points[i].rho);
    const dot = figma.createEllipse();
    dot.resize(ptR * 2, ptR * 2);
    dot.x = px.px - ptR;
    dot.y = px.py - ptR;
    dot.fills = [solidFill(PALETTE.red, 0.85)];
    dot.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.4 }];
    dot.strokeWeight = 0.4;
    frame.appendChild(dot);
  }

  addText(frame, panel.x_label || "", x0, y1 + 18, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yl = addText(frame, panel.y_label || "", 2, y0 + plotH * 0.25, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
  });
  yl.rotation = -90;

  return frame;
}

/**
 * Resolution sensitivity panel: 0.5 Å bin sweep (median ρ vs bin center).
 * Matches qscore_resolution_sensitivity.py panel b.
 */
function buildResolutionSweepPanel(panel, bodyFont, monoFont) {
  const points = panel.points || [];
  const W = 360;
  const H = 280;
  const padL = 52;
  const padR = 16;
  const padT = 38;
  const padB = 44;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xs = points.map(function (p) { return p.x; });
  const ys = points.map(function (p) { return p.y; });
  const xb = axisBounds(xs, 0.04);
  const yb = axisBounds(ys, 0.12);
  const cutoff = panel.cutoff_a != null ? panel.cutoff_a : 4.0;

  const frame = figma.createFrame();
  frame.name = "resolution sweep: 0.5 Å bins";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.letter || "b", 8, 6, bodyFont, { size: 12, color: PALETTE.muted });
  addText(frame, panel.title || "", 24, 6, bodyFont, { size: 9.5, color: PALETTE.ink, width: W - 28 });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  function dataToPx(x, y) {
    const tx = (x - xb.min) / (xb.max - xb.min);
    const ty = (y - yb.min) / (yb.max - yb.min);
    return { px: x0 + tx * plotW, py: y1 - ty * plotH };
  }

  for (let g = 1; g < 4; g++) {
    addLine(frame, x0 + (plotW * g) / 4, y0, x0 + (plotW * g) / 4, y1, { color: PALETTE.grid, weight: 0.5 });
    addLine(frame, x0, y0 + (plotH * g) / 4, x1, y0 + (plotH * g) / 4, { color: PALETTE.grid, weight: 0.5 });
  }
  addLine(frame, x0, y1, x1, y1, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0, y0, x0, y1, { color: PALETTE.ink, weight: 1 });

  const xTicks = pickTicks(xb.min, xb.max, 5);
  const yTicks = pickTicks(yb.min, yb.max, 5);
  drawXAxisTicks(frame, x0, y1, plotW, xb, xTicks, monoFont, "resolution");
  drawYAxisTicks(frame, x0, y0, plotH, yb, yTicks, monoFont, "rho");

  if (yb.min <= 0 && yb.max >= 0) {
    const zy = dataToPx(xb.min, 0).py;
    addLine(frame, x0, zy, x1, zy, { color: PALETTE.muted, weight: 0.6, opacity: 0.7 });
  }
  if (cutoff >= xb.min && cutoff <= xb.max) {
    const cx = dataToPx(cutoff, yb.min).px;
    addLine(frame, cx, y0, cx, y1, { color: PALETTE.muted, weight: 0.8, dash: [4, 3], opacity: 0.85 });
    addText(frame, panel.cutoff_label || "4 Å cutoff", cx + 3, y0 + 2, bodyFont, {
      size: 6.5,
      color: PALETTE.muted,
    });
  }

  if (points.length >= 2) {
    for (let i = 1; i < points.length; i++) {
      const p0 = dataToPx(points[i - 1].x, points[i - 1].y);
      const p1 = dataToPx(points[i].x, points[i].y);
      addLine(frame, p0.px, p0.py, p1.px, p1.py, { color: PALETTE.blue, weight: 1.2 });
    }
  }

  const ptR = 4.5;
  for (let i = 0; i < points.length; i++) {
    const px = dataToPx(points[i].x, points[i].y);
    const dot = figma.createEllipse();
    dot.resize(ptR * 2, ptR * 2);
    dot.x = px.px - ptR;
    dot.y = px.py - ptR;
    dot.fills = [solidFill(PALETTE.blue, 0.95)];
    dot.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
    dot.strokeWeight = 0.4;
    frame.appendChild(dot);
  }

  addText(frame, panel.x_label || "", x0, y1 + 18, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yl = addText(frame, panel.y_label || "", 2, y0 + plotH * 0.22, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
  });
  yl.rotation = -90;

  return frame;
}

/**
 * Resolution sensitivity panel A: median ρ in standard cohort bins (vertical bars).
 */
function buildStandardBinsPanel(panel, bodyFont, monoFont) {
  const bars = panel.bars || [];
  const n = bars.length;
  const W = 280;
  const H = 260;
  const padL = 52;
  const padR = 14;
  const padT = 38;
  const padB = 52;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const ys = bars.map(function (b) { return b.median_rho; });
  const yb = axisBounds(ys, 0.15);
  if (yb.min > 0) yb.min = 0;
  if (yb.max < 0) yb.max = 0;

  const frame = figma.createFrame();
  frame.name = "resolution sensitivity: standard bins";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.letter || "a", 8, 6, bodyFont, { size: 12, color: PALETTE.muted });
  addText(frame, panel.title || "", 24, 6, bodyFont, { size: 9.5, color: PALETTE.ink, width: W - 28 });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  function rhoToY(rho) {
    return y1 - ((rho - yb.min) / (yb.max - yb.min)) * plotH;
  }

  addLine(frame, x0, y1, x1, y1, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0, y0, x0, y1, { color: PALETTE.ink, weight: 1 });

  const yTicks = pickTicks(yb.min, yb.max, 5);
  drawYAxisTicks(frame, x0, y0, plotH, yb, yTicks, monoFont, "rho");

  if (yb.min <= 0 && yb.max >= 0) {
    const zy = rhoToY(0);
    addLine(frame, x0, zy, x1, zy, { color: PALETTE.muted, weight: 0.6, opacity: 0.7 });
  }

  const gap = 8;
  const barW = n > 0 ? (plotW - gap * (n - 1)) / n : plotW;

  for (let i = 0; i < n; i++) {
    const b = bars[i];
    const bx = x0 + i * (barW + gap);
    const yTop = rhoToY(Math.max(0, b.median_rho));
    const yBase = rhoToY(Math.min(0, b.median_rho));
    const barH = Math.max(1, Math.abs(yBase - yTop));

    const rect = figma.createRectangle();
    rect.resize(barW, barH);
    rect.x = bx;
    rect.y = Math.min(yTop, yBase);
    rect.fills = [solidFill(PALETTE.red, 0.88)];
    rect.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
    rect.strokeWeight = 0.4;
    rect.cornerRadius = 1;
    frame.appendChild(rect);

    const lbl = b.label || "";
    addText(frame, lbl, bx - 2, y1 + 4, bodyFont, {
      size: 6.5,
      color: PALETTE.muted,
      width: barW + 4,
      align: "CENTER",
    });

    const nLabel = "n=" + b.n;
    const tipY = Math.min(yTop, yBase) - 10;
    addText(frame, nLabel, bx, tipY, monoFont, {
      size: 6,
      color: PALETTE.ink,
      width: barW + 4,
      align: "CENTER",
    });
  }

  addText(frame, panel.x_label || "", x0, y1 + 28, bodyFont, {
    size: 7,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yl = addText(frame, panel.y_label || "", 2, y0 + plotH * 0.2, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
  });
  yl.rotation = -90;

  return frame;
}

/**
 * Resolution sensitivity panel C: median ρ vs resolution ceiling (≤ vs > cutoff).
 */
function buildCutoffSensitivityPanel(panel, bodyFont, monoFont) {
  const seriesLe = panel.series_le || [];
  const seriesGt = panel.series_gt || [];
  const W = 340;
  const H = 280;
  const padL = 52;
  const padR = 16;
  const padT = 38;
  const padB = 44;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const allY = seriesLe.map(function (p) { return p.y; }).concat(seriesGt.map(function (p) { return p.y; }));
  const allX = seriesLe.map(function (p) { return p.x; }).concat(seriesGt.map(function (p) { return p.x; }));
  const xb = axisBounds(allX, 0.06);
  const yb = axisBounds(allY, 0.12);
  const cutoff = panel.cutoff_a != null ? panel.cutoff_a : 4.0;

  const frame = figma.createFrame();
  frame.name = "resolution sensitivity: cutoff";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.letter || "c", 8, 6, bodyFont, { size: 12, color: PALETTE.muted });
  addText(frame, panel.title || "", 24, 6, bodyFont, { size: 9.5, color: PALETTE.ink, width: W - 28 });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  function dataToPx(x, y) {
    const tx = (x - xb.min) / (xb.max - xb.min);
    const ty = (y - yb.min) / (yb.max - yb.min);
    return { px: x0 + tx * plotW, py: y1 - ty * plotH };
  }

  for (let g = 1; g < 4; g++) {
    addLine(frame, x0 + (plotW * g) / 4, y0, x0 + (plotW * g) / 4, y1, { color: PALETTE.grid, weight: 0.5 });
    addLine(frame, x0, y0 + (plotH * g) / 4, x1, y0 + (plotH * g) / 4, { color: PALETTE.grid, weight: 0.5 });
  }
  addLine(frame, x0, y1, x1, y1, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0, y0, x0, y1, { color: PALETTE.ink, weight: 1 });

  const xTicks = pickTicks(xb.min, xb.max, 5);
  const yTicks = pickTicks(yb.min, yb.max, 5);
  drawXAxisTicks(frame, x0, y1, plotW, xb, xTicks, monoFont, "resolution");
  drawYAxisTicks(frame, x0, y0, plotH, yb, yTicks, monoFont, "rho");

  if (yb.min <= 0 && yb.max >= 0) {
    const zy = dataToPx(xb.min, 0).py;
    addLine(frame, x0, zy, x1, zy, { color: PALETTE.muted, weight: 0.6, opacity: 0.7 });
  }
  if (cutoff >= xb.min && cutoff <= xb.max) {
    const cx = dataToPx(cutoff, yb.min).px;
    addLine(frame, cx, y0, cx, y1, { color: PALETTE.muted, weight: 0.7, dash: [2, 2], opacity: 0.75 });
  }

  function drawSeries(points, color, dash, marker) {
    if (points.length >= 2) {
      for (let i = 1; i < points.length; i++) {
        const p0 = dataToPx(points[i - 1].x, points[i - 1].y);
        const p1 = dataToPx(points[i].x, points[i].y);
        addLine(frame, p0.px, p0.py, p1.px, p1.py, {
          color: color,
          weight: marker === "square" ? 1 : 1.2,
          dash: dash,
        });
      }
    }
    const r = marker === "square" ? 3.5 : 4;
    for (let i = 0; i < points.length; i++) {
      const px = dataToPx(points[i].x, points[i].y);
      if (marker === "square") {
        const sq = figma.createRectangle();
        sq.resize(r * 2, r * 2);
        sq.x = px.px - r;
        sq.y = px.py - r;
        sq.fills = [solidFill(color, 0.9)];
        sq.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
        sq.strokeWeight = 0.4;
        sq.cornerRadius = 0.5;
        frame.appendChild(sq);
      } else {
        const dot = figma.createEllipse();
        dot.resize(r * 2, r * 2);
        dot.x = px.px - r;
        dot.y = px.py - r;
        dot.fills = [solidFill(color, 0.9)];
        dot.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
        dot.strokeWeight = 0.4;
        frame.appendChild(dot);
      }
    }
  }

  drawSeries(seriesLe, PALETTE.red, null, "circle");
  drawSeries(seriesGt, PALETTE.magenta, [4, 3], "square");

  const legX = x0 + plotW - 78;
  const legY = y0 + 4;
  addLine(frame, legX, legY + 4, legX + 14, legY + 4, { color: PALETTE.red, weight: 1.2 });
  const legDot = figma.createEllipse();
  legDot.resize(6, 6);
  legDot.x = legX + 4;
  legDot.y = legY + 1;
  legDot.fills = [solidFill(PALETTE.red)];
  legDot.strokes = [];
  frame.appendChild(legDot);
  addText(frame, panel.legend_le || "res ≤ cutoff", legX + 18, legY, bodyFont, {
    size: 6.5,
    color: PALETTE.ink,
    width: 58,
  });

  const legY2 = legY + 14;
  addLine(frame, legX, legY2 + 4, legX + 14, legY2 + 4, {
    color: PALETTE.magenta,
    weight: 1,
    dash: [3, 2],
  });
  const legSq = figma.createRectangle();
  legSq.resize(5, 5);
  legSq.x = legX + 5;
  legSq.y = legY2 + 2;
  legSq.fills = [solidFill(PALETTE.magenta)];
  legSq.strokes = [];
  frame.appendChild(legSq);
  addText(frame, panel.legend_gt || "res > cutoff", legX + 18, legY2, bodyFont, {
    size: 6.5,
    color: PALETTE.ink,
    width: 58,
  });

  addText(frame, panel.x_label || "", x0, y1 + 18, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yl = addText(frame, panel.y_label || "", 2, y0 + plotH * 0.22, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
  });
  yl.rotation = -90;

  return frame;
}

function buildResolutionSensitivityFigure(data, bodyFont, monoFont, includeCutoff) {
  const panelA = buildStandardBinsPanel(data.panels.resolution_standard_bins, bodyFont, monoFont);
  const panelB = buildResolutionSweepPanel(data.panels.resolution_sweep, bodyFont, monoFont);
  const panelC = includeCutoff && data.panels.resolution_cutoff
    ? buildCutoffSensitivityPanel(data.panels.resolution_cutoff, bodyFont, monoFont)
    : null;

  const gap = 16;
  const margin = 16;
  const wrapper = figma.createFrame();
  wrapper.name = includeCutoff
    ? "ρ(Q,V) resolution sensitivity (A+B+C)"
    : "ρ(Q,V) resolution sensitivity (A+B)";
  wrapper.layoutMode = "NONE";
  wrapper.clipsContent = false;
  wrapper.fills = [solidFill({ r: 1, g: 1, b: 1 })];
  wrapper.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 8;

  let totalW = 0;
  const panels = [panelA, panelB];
  if (panelC) panels.push(panelC);

  let y = margin;
  for (let i = 0; i < panels.length; i++) {
    totalW += panels[i].width;
    if (i < panels.length - 1) totalW += gap;
  }

  const titleText = data.sensitivity_title || "ρ(Q, V) resolution sensitivity";
  const title = addText(wrapper, titleText, margin, y, bodyFont, {
    size: 12,
    color: PALETTE.ink,
    width: totalW,
  });
  y += title.height + 10;

  let cursorX = margin;
  let contentH = 0;
  for (let i = 0; i < panels.length; i++) {
    panels[i].x = cursorX;
    panels[i].y = y;
    wrapper.appendChild(panels[i]);
    cursorX += panels[i].width + gap;
    contentH = Math.max(contentH, panels[i].height);
  }

  wrapper.resize(margin + totalW + margin, y + contentH + margin);
  return wrapper;
}

function buildCohortFigure(data, bodyFont, monoFont) {
  const panelA = buildCohortPanelA(data.panels.a, bodyFont, monoFont);
  const panelB = buildCohortPanelB(data.panels.b, bodyFont, monoFont);

  const gap = 16;
  const margin = 16;
  const wrapper = figma.createFrame();
  wrapper.name = "Q-score vs V — cohort summary";
  wrapper.layoutMode = "NONE";
  wrapper.clipsContent = false;
  wrapper.fills = [solidFill({ r: 1, g: 1, b: 1 })];
  wrapper.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 8;

  let y = margin;
  if (data.figure_title) {
    const title = addText(wrapper, data.figure_title, margin, y, bodyFont, {
      size: 13,
      color: PALETTE.ink,
      width: panelA.width + gap + panelB.width,
    });
    y += title.height + 10;
  }

  panelA.x = margin;
  panelA.y = y;
  wrapper.appendChild(panelA);
  panelB.x = margin + panelA.width + gap;
  panelB.y = y;
  wrapper.appendChild(panelB);

  const contentH = Math.max(panelA.height, panelB.height);
  const totalW = margin + panelA.width + gap + panelB.width + margin;
  const totalH = y + contentH + margin;
  wrapper.resize(totalW, totalH);

  return wrapper;
}

figma.ui.onmessage = async function (msg) {
  try {
    const bodyFont = await loadBodyFont();
    const monoFont = await loadMonoFont();
    const data = msg.exportData;

    if (!data || !data.panels) {
      throw new Error(
        "Missing cohort export. Run: uv run python scripts/run_qscore_figma_export.py"
      );
    }

    let node;
    if (msg.mode === "panel_a") {
      node = buildCohortPanelA(data.panels.a, bodyFont, monoFont);
    } else if (msg.mode === "panel_b") {
      node = buildCohortPanelB(data.panels.b, bodyFont, monoFont);
    } else if (msg.mode === "resolution_standard_bins") {
      if (!data.panels.resolution_standard_bins) {
        throw new Error("Missing standard bins data — re-run export script.");
      }
      node = buildStandardBinsPanel(data.panels.resolution_standard_bins, bodyFont, monoFont);
    } else if (msg.mode === "resolution_sweep") {
      if (!data.panels.resolution_sweep) {
        throw new Error("Missing resolution sweep data — re-run export script.");
      }
      node = buildResolutionSweepPanel(data.panels.resolution_sweep, bodyFont, monoFont);
    } else if (msg.mode === "resolution_sensitivity_ab") {
      if (!data.panels.resolution_standard_bins || !data.panels.resolution_sweep) {
        throw new Error("Missing resolution sensitivity data — re-run export script.");
      }
      node = buildResolutionSensitivityFigure(data, bodyFont, monoFont, false);
    } else if (msg.mode === "resolution_cutoff") {
      if (!data.panels.resolution_cutoff) {
        throw new Error("Missing cutoff sensitivity data — re-run export script.");
      }
      node = buildCutoffSensitivityPanel(data.panels.resolution_cutoff, bodyFont, monoFont);
    } else if (msg.mode === "resolution_sensitivity_abc") {
      if (!data.panels.resolution_standard_bins || !data.panels.resolution_sweep || !data.panels.resolution_cutoff) {
        throw new Error("Missing resolution sensitivity data — re-run export script.");
      }
      node = buildResolutionSensitivityFigure(data, bodyFont, monoFont, true);
    } else {
      node = buildCohortFigure(data, bodyFont, monoFont);
    }

    const origin = findClearOrigin();
    node.x = origin.x;
    node.y = origin.y;
    figma.currentPage.appendChild(node);
    figma.currentPage.selection = [node];
    figma.viewport.scrollAndZoomIntoView([node]);
    figma.ui.postMessage({ type: "result", text: "Created \"" + node.name + "\"." });
  } catch (err) {
    figma.ui.postMessage({
      type: "error",
      text: err instanceof Error ? err.message : String(err),
    });
  }
};
