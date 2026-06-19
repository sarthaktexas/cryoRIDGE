/**
 * Thesis Cross Metric — Figma plugin
 *
 * Recreates cohort cross-metric figures:
 *   - median Spearman ρ heatmap (metric × metric)
 *   - grouped horizontal bars: per-map ρ vs BlocRes for key pairs
 */

figma.showUI(__html__, { width: 300, height: 420 });

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
  green: hexToRgb("#3BBF6A"),
  magenta: hexToRgb("#BA3EC3"),
  white: { r: 1, g: 1, b: 1 },
  ink: { r: 0.12, g: 0.12, b: 0.14 },
  muted: { r: 0.42, g: 0.42, b: 0.46 },
  grid: { r: 0.88, g: 0.88, b: 0.9 },
  panelBg: { r: 1, g: 1, b: 1 },
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

let cachedBodyFont = null;
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
  if (!isFinite(min)) return { min: -1, max: 1 };
  if (min === max) {
    const d = Math.abs(min) * 0.1 || 0.1;
    return { min: min - d, max: max + d };
  }
  const span = max - min;
  return { min: min - span * padFrac, max: max + span * padFrac };
}

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

function formatTick(v) {
  if (Math.abs(v) < 1e-10) return "0";
  return v.toFixed(2);
}

function divergingRgb(t, vmin, vmax) {
  const x = Math.max(vmin, Math.min(vmax, t));
  const norm = (x - vmin) / (vmax - vmin);
  if (norm <= 0.5) {
    const f = norm / 0.5;
    return {
      r: PALETTE.blue.r + (PALETTE.white.r - PALETTE.blue.r) * f,
      g: PALETTE.blue.g + (PALETTE.white.g - PALETTE.blue.g) * f,
      b: PALETTE.blue.b + (PALETTE.white.b - PALETTE.blue.b) * f,
    };
  }
  const f = (norm - 0.5) / 0.5;
  return {
    r: PALETTE.white.r + (PALETTE.red.r - PALETTE.white.r) * f,
    g: PALETTE.white.g + (PALETTE.red.g - PALETTE.white.g) * f,
    b: PALETTE.white.b + (PALETTE.red.b - PALETTE.white.b) * f,
  };
}

function textColorForCell(val, vmin, vmax) {
  const norm = (val - vmin) / (vmax - vmin);
  return norm > 0.35 && norm < 0.65 ? PALETTE.ink : PALETTE.white;
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

function drawXAxisTicks(frame, x0, yBase, plotW, bounds, ticks, monoFont) {
  const tickLen = 4;
  for (let i = 0; i < ticks.length; i++) {
    const t = ticks[i];
    const tx = (t - bounds.min) / (bounds.max - bounds.min);
    const px = x0 + tx * plotW;
    addLine(frame, px, yBase, px, yBase + tickLen, { color: PALETTE.ink, weight: 0.6 });
    addText(frame, formatTick(t), px - 14, yBase + tickLen + 1, monoFont, {
      size: 6.5,
      color: PALETTE.muted,
      width: 28,
      align: "CENTER",
    });
  }
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

function addDivergingColorbar(parent, x, y, w, h, vmin, vmax, label, bodyFont) {
  const nSteps = 24;
  const stepH = h / nSteps;
  for (let i = 0; i < nSteps; i++) {
    const t = vmax - (i / Math.max(nSteps - 1, 1)) * (vmax - vmin);
    const rect = figma.createRectangle();
    rect.resize(w, stepH + 0.5);
    rect.x = x;
    rect.y = y + i * stepH;
    rect.fills = [solidFill(divergingRgb(t, vmin, vmax))];
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
  addText(parent, "+1", x + w + 3, y - 1, bodyFont, { size: 6.5, color: PALETTE.muted });
  addText(parent, "0", x + w + 3, y + h * 0.5 - 4, bodyFont, { size: 6.5, color: PALETTE.muted });
  addText(parent, "−1", x + w + 3, y + h - 7, bodyFont, { size: 6.5, color: PALETTE.muted });
  const lbl = addText(parent, label, x + w + 14, y + h * 0.25, bodyFont, {
    size: 7,
    color: PALETTE.muted,
    width: 80,
  });
  lbl.rotation = -90;
}

function buildMedianHeatmap(panel, bodyFont, monoFont) {
  const rows = panel.row_labels || [];
  const cols = panel.col_labels || [];
  const n = rows.length;
  const vmin = panel.vmin != null ? panel.vmin : -1;
  const vmax = panel.vmax != null ? panel.vmax : 1;
  const cellSize = 36;
  const labelH = 52;
  const labelW = 72;
  const padT = 38;
  const padR = 72;
  const padB = 16;
  const W = labelW + n * cellSize + padR;
  const H = padT + labelH + n * cellSize + padB;

  const frame = figma.createFrame();
  frame.name = "cross metric: median heatmap";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.title || "", 8, 6, bodyFont, { size: 9.5, color: PALETTE.ink, width: W - 16 });

  const x0 = labelW;
  const y0 = padT + labelH;

  for (let j = 0; j < n; j++) {
    const lbl = cols[j] || "";
    const shortLbl = lbl.length > 14 ? lbl.slice(0, 12) + "…" : lbl;
    const tx = addText(frame, shortLbl, x0 + j * cellSize + 2, padT + 8, bodyFont, {
      size: 6.5,
      color: PALETTE.ink,
      width: cellSize + 20,
    });
    tx.rotation = -45;
  }

  for (let i = 0; i < n; i++) {
    const lbl = rows[i] || "";
    const shortLbl = lbl.length > 12 ? lbl.slice(0, 10) + "…" : lbl;
    addText(frame, shortLbl, 4, y0 + i * cellSize + 12, bodyFont, {
      size: 6.5,
      color: PALETTE.ink,
      width: labelW - 8,
      align: "RIGHT",
    });
  }

  const cells = panel.cells || [];
  for (let k = 0; k < cells.length; k++) {
    const c = cells[k];
    const rect = figma.createRectangle();
    rect.resize(cellSize - 1, cellSize - 1);
    rect.x = x0 + c.col * cellSize;
    rect.y = y0 + c.row * cellSize;
    rect.fills = [solidFill(divergingRgb(c.value, vmin, vmax))];
    rect.strokes = [{ type: "SOLID", color: { r: 0.85, g: 0.85, b: 0.87 }, opacity: 0.8 }];
    rect.strokeWeight = 0.4;
    frame.appendChild(rect);

    const tc = textColorForCell(c.value, vmin, vmax);
    addText(frame, c.label || "", rect.x + 4, rect.y + 12, monoFont, {
      size: 7,
      color: tc,
      width: cellSize - 6,
      align: "CENTER",
    });
  }

  addDivergingColorbar(
    frame,
    x0 + n * cellSize + 8,
    y0,
    10,
    Math.min(n * cellSize, 160),
    vmin,
    vmax,
    panel.colorbar_label || "Median Spearman ρ",
    bodyFont
  );

  return frame;
}

function buildLocresPairsPanel(panel, bodyFont, monoFont) {
  const structures = panel.structures || [];
  const series = panel.series || [];
  const n = structures.length;
  const labelW = 118;
  const padT = 38;
  const padB = 40;
  const padR = 14;
  const barH = 4;
  const groupH = 14;
  const groupGap = 4;
  const plotH = n * (groupH + groupGap) - groupGap;
  const W = 520;
  const H = padT + plotH + padB;
  const plotW = W - labelW - padR - 18;
  const padL = labelW + 8;

  const allRhos = [];
  for (let s = 0; s < series.length; s++) {
    const vals = series[s].values || [];
    for (let i = 0; i < vals.length; i++) {
      if (vals[i].rho != null && isFinite(vals[i].rho)) allRhos.push(vals[i].rho);
    }
  }
  const xb = axisBounds(allRhos, 0.05);
  if (xb.min > -0.05) xb.min = Math.min(xb.min, -0.1);
  if (xb.max < 0.05) xb.max = Math.max(xb.max, 0.1);

  const frame = figma.createFrame();
  frame.name = "cross metric: locres pairs";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.title || "", 8, 6, bodyFont, { size: 9.5, color: PALETTE.ink, width: W - 16 });

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

  addLine(frame, x0, y1 + 4, x1, y1 + 4, { color: PALETTE.ink, weight: 1 });
  drawXAxisTicks(frame, x0, y1 + 4, plotW, xb, pickTicks(xb.min, xb.max, 5), monoFont);

  const width = barH;
  const offsets = series.map(function (s) {
    return (s.offset_index != null ? s.offset_index : 0) * width;
  });

  for (let i = 0; i < n; i++) {
    const struct = structures[i];
    const gy = y0 + i * (groupH + groupGap);
    const lbl = struct.label || struct.emdb_id || "";
    const shortLbl = lbl.length > 22 ? lbl.slice(0, 20) + "…" : lbl;
    addText(frame, shortLbl, 4, gy + 3, bodyFont, {
      size: 6.5,
      color: PALETTE.ink,
      width: labelW - 6,
      align: "RIGHT",
    });

    for (let s = 0; s < series.length; s++) {
      const ser = series[s];
      const vals = ser.values || [];
      const pt = vals[i];
      if (!pt || pt.rho == null || !isFinite(pt.rho)) continue;
      const color = hexToRgb(ser.color || "#888888");
      const by = gy + groupH / 2 + offsets[s] - width / 2;
      const xStart = rhoToX(Math.min(0, pt.rho));
      const xEnd = rhoToX(Math.max(0, pt.rho));
      const bw = Math.max(1, Math.abs(xEnd - xStart));
      const bx = Math.min(xStart, xEnd);

      const rect = figma.createRectangle();
      rect.resize(bw, width);
      rect.x = bx;
      rect.y = by;
      rect.fills = [solidFill(color, 0.92)];
      rect.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
      rect.strokeWeight = 0.3;
      rect.cornerRadius = 1;
      frame.appendChild(rect);
    }
  }

  addText(frame, panel.x_label || "", x0, y1 + 22, bodyFont, {
    size: 7.5,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });

  let legX = x1 - 140;
  let legY = y0 + 4;
  for (let s = 0; s < series.length; s++) {
    const ser = series[s];
    const color = hexToRgb(ser.color || "#888888");
    const sw = figma.createRectangle();
    sw.resize(10, 4);
    sw.x = legX;
    sw.y = legY;
    sw.fills = [solidFill(color)];
    sw.strokes = [];
    frame.appendChild(sw);
    addText(frame, ser.label || ser.key || "", legX + 14, legY - 2, bodyFont, {
      size: 6.5,
      color: PALETTE.ink,
      width: 120,
    });
    legY += 12;
  }

  return frame;
}

function buildCombinedFigure(data, bodyFont, monoFont) {
  const heat = buildMedianHeatmap(data.panels.median_heatmap, bodyFont, monoFont);
  const locres = buildLocresPairsPanel(data.panels.locres_pairs, bodyFont, monoFont);

  const wrapper = figma.createFrame();
  wrapper.name = data.figure_title || "cross metric cohort";
  wrapper.layoutMode = "HORIZONTAL";
  wrapper.primaryAxisSizingMode = "AUTO";
  wrapper.counterAxisSizingMode = "AUTO";
  wrapper.itemSpacing = 24;
  wrapper.paddingLeft = 16;
  wrapper.paddingRight = 16;
  wrapper.paddingTop = 16;
  wrapper.paddingBottom = 16;
  wrapper.fills = [solidFill(PALETTE.panelBg)];
  wrapper.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 8;
  wrapper.appendChild(heat);
  wrapper.appendChild(locres);
  return wrapper;
}

figma.ui.onmessage = async function (msg) {
  if (msg.type !== "generate") return;

  try {
    const bodyFont = await loadBodyFont();
    const monoFont = await loadMonoFont();
    const data = msg.data;
    if (!data || !data.panels) throw new Error("Missing export data.");

    let node;
    if (msg.mode === "median_heatmap") {
      if (!data.panels.median_heatmap) throw new Error("Missing median heatmap data.");
      node = buildMedianHeatmap(data.panels.median_heatmap, bodyFont, monoFont);
    } else if (msg.mode === "locres_pairs") {
      if (!data.panels.locres_pairs) throw new Error("Missing locres pairs data.");
      node = buildLocresPairsPanel(data.panels.locres_pairs, bodyFont, monoFont);
    } else if (msg.mode === "both") {
      node = buildCombinedFigure(data, bodyFont, monoFont);
    } else {
      throw new Error("Unknown mode: " + msg.mode);
    }

    const origin = findClearOrigin();
    node.x = origin.x;
    node.y = origin.y;
    figma.currentPage.appendChild(node);
    figma.viewport.scrollAndZoomIntoView([node]);
    figma.ui.postMessage({ type: "done", name: node.name });
  } catch (err) {
    figma.ui.postMessage({ type: "error", message: String(err.message || err) });
  }
};
