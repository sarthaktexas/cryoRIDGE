/**
 * Thesis Placement Utility — Figma plugin
 *
 * Recreates placement validation figures:
 *   - head-to-head: enrichment / classification / AUC (3 horizontal bar panels)
 *   - rank recovery: median ρ(Q, proxy) vertical bars
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
  if (!isFinite(min)) return { min: 0, max: 1 };
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
  if (Math.abs(v) >= 10) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(1);
  return v.toFixed(2);
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

function drawYAxisTicks(frame, xBase, y0, plotH, bounds, ticks, monoFont) {
  const tickLen = 4;
  const y1 = y0 + plotH;
  for (let i = 0; i < ticks.length; i++) {
    const t = ticks[i];
    const ty = (t - bounds.min) / (bounds.max - bounds.min);
    const py = y1 - ty * plotH;
    addLine(frame, xBase - tickLen, py, xBase, py, { color: PALETTE.ink, weight: 0.6 });
    addText(frame, formatTick(t), xBase - tickLen - 22, py - 4, monoFont, {
      size: 6.5,
      color: PALETTE.muted,
      width: 20,
      align: "RIGHT",
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

function buildHeadToHeadSubpanel(panelDef, predictors, bodyFont, monoFont, showLabels) {
  const metric = panelDef.metric;
  const n = predictors.length;
  const labelW = showLabels ? 108 : 8;
  const padT = 32;
  const padB = 36;
  const padR = 10;
  const barH = 12;
  const barGap = 3;
  const plotH = n * (barH + barGap) - barGap;
  const W = showLabels ? 220 : 130;
  const H = padT + plotH + padB;
  const plotW = W - labelW - padR - 12;
  const padL = labelW + 6;

  const vals = predictors.map(function (p) { return p[metric]; });
  const xb = axisBounds(vals, 0.05);
  if (xb.min > 0) xb.min = 0;

  const frame = figma.createFrame();
  frame.name = "head-to-head: " + (panelDef.title || metric);
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panelDef.title || "", 8, 6, bodyFont, {
    size: 8.5,
    color: PALETTE.ink,
    width: W - 12,
  });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  function valToX(v) {
    return x0 + ((v - xb.min) / (xb.max - xb.min)) * plotW;
  }

  addLine(frame, x0, y1 + 4, x1, y1 + 4, { color: PALETTE.ink, weight: 1 });
  drawXAxisTicks(frame, x0, y1 + 4, plotW, xb, pickTicks(xb.min, xb.max, 4), monoFont);

  for (let i = 0; i < n; i++) {
    const p = predictors[i];
    const v = p[metric];
    const by = y0 + i * (barH + barGap);
    const color = hexToRgb(p.color || "#888888");
    const bx = x0;
    const bw = Math.max(1, valToX(v) - x0);

    const rect = figma.createRectangle();
    rect.resize(bw, barH);
    rect.x = bx;
    rect.y = by;
    rect.fills = [solidFill(color, 0.92)];
    rect.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
    rect.strokeWeight = 0.4;
    rect.cornerRadius = 1;
    frame.appendChild(rect);

    if (showLabels) {
      const lbl = p.label || p.predictor || "";
      const shortLbl = lbl.length > 20 ? lbl.slice(0, 18) + "…" : lbl;
      addText(frame, shortLbl, 4, by + 2, bodyFont, {
        size: 6.5,
        color: PALETTE.ink,
        width: labelW - 6,
        align: "RIGHT",
      });
    }
  }

  addText(frame, panelDef.x_label || "", x0, y1 + 20, bodyFont, {
    size: 6.5,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });

  return frame;
}

function buildHeadToHeadFigure(panel, bodyFont, monoFont) {
  const predictors = panel.predictors || [];
  const subpanels = panel.panels || [];
  const row = figma.createFrame();
  row.name = "head-to-head panels";
  row.layoutMode = "HORIZONTAL";
  row.primaryAxisSizingMode = "AUTO";
  row.counterAxisSizingMode = "AUTO";
  row.itemSpacing = 12;
  row.fills = [];
  row.strokes = [];
  for (let i = 0; i < subpanels.length; i++) {
    row.appendChild(buildHeadToHeadSubpanel(subpanels[i], predictors, bodyFont, monoFont, i === 0));
  }

  const wrapper = figma.createFrame();
  wrapper.name = panel.title || "placement head-to-head";
  wrapper.layoutMode = "VERTICAL";
  wrapper.primaryAxisSizingMode = "AUTO";
  wrapper.counterAxisSizingMode = "AUTO";
  wrapper.itemSpacing = 8;
  wrapper.paddingLeft = 12;
  wrapper.paddingRight = 12;
  wrapper.paddingTop = 12;
  wrapper.paddingBottom = 12;
  wrapper.fills = [solidFill(PALETTE.panelBg)];
  wrapper.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 8;

  const title = addText(wrapper, panel.title || "", 0, 0, bodyFont, {
    size: 10,
    color: PALETTE.ink,
    width: 640,
  });
  title.layoutAlign = "STRETCH";
  wrapper.appendChild(row);
  return wrapper;
}

function buildRankRecoveryPanel(panel, bodyFont, monoFont) {
  const bars = panel.bars || [];
  const n = bars.length;
  const W = 340;
  const H = 260;
  const padL = 48;
  const padR = 14;
  const padT = 38;
  const padB = 52;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const vals = bars.map(function (b) { return b.median_rho; }).filter(function (v) { return v != null && isFinite(v); });
  const yb = axisBounds(vals.length ? vals : [-1, 1], 0.1);
  if (yb.min > -0.05) yb.min = Math.min(yb.min, -0.15);
  if (yb.max < 0.05) yb.max = Math.max(yb.max, 0.15);

  const frame = figma.createFrame();
  frame.name = "placement rank recovery";
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

  function valToY(v) {
    return y1 - ((v - yb.min) / (yb.max - yb.min)) * plotH;
  }

  if (yb.min <= 0 && yb.max >= 0) {
    const zy = valToY(0);
    addLine(frame, x0 - 2, zy, x1 + 2, zy, { color: PALETTE.muted, weight: 0.6, opacity: 0.7 });
  }

  addLine(frame, x0 - 4, y0, x0 - 4, y1 + 4, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0 - 4, y1 + 4, x1, y1 + 4, { color: PALETTE.ink, weight: 1 });
  drawYAxisTicks(frame, x0 - 4, y0, plotH, yb, pickTicks(yb.min, yb.max, 5), monoFont);

  const barW = Math.min(36, plotW / Math.max(n, 1) - 8);
  const gap = (plotW - n * barW) / Math.max(n + 1, 1);

  for (let i = 0; i < n; i++) {
    const b = bars[i];
    if (b.median_rho == null || !isFinite(b.median_rho)) continue;
    const color = hexToRgb(b.color || "#888888");
    const bx = x0 + gap + i * (barW + gap);
    const zy = valToY(0);
    const topY = valToY(b.median_rho);
    const by = Math.min(zy, topY);
    const bh = Math.max(1, Math.abs(topY - zy));

    const rect = figma.createRectangle();
    rect.resize(barW, bh);
    rect.x = bx;
    rect.y = by;
    rect.fills = [solidFill(color, 0.92)];
    rect.strokes = [{ type: "SOLID", color: { r: 0.2, g: 0.2, b: 0.2 }, opacity: 0.35 }];
    rect.strokeWeight = 0.4;
    rect.cornerRadius = 1;
    frame.appendChild(rect);

    const lbl = b.label || "";
    const shortLbl = lbl.length > 10 ? lbl.slice(0, 9) + "…" : lbl;
    const tx = addText(frame, shortLbl, bx - 4, y1 + 8, bodyFont, {
      size: 6.5,
      color: PALETTE.ink,
      width: barW + 12,
      align: "CENTER",
    });
    tx.rotation = -20;
  }

  const yl = addText(frame, panel.y_label || "", 4, y0 + plotH * 0.35, bodyFont, {
    size: 7,
    color: PALETTE.muted,
    width: 80,
  });
  yl.rotation = -90;

  return frame;
}

function buildLowQRocPanel(panel, bodyFont, monoFont) {
  const curves = panel.curves || [];
  const W = 320;
  const H = 300;
  const padL = 48;
  const padR = 14;
  const padT = 38;
  const padB = 56;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const xb = { min: 0, max: 1 };
  const yb = { min: 0, max: 1 };

  const frame = figma.createFrame();
  frame.name = "placement low-Q ROC";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  addText(frame, panel.title || "Low-Q ROC", 8, 6, bodyFont, {
    size: 8.5,
    color: PALETTE.ink,
    width: W - 16,
  });

  const x0 = padL;
  const y0 = padT;
  const x1 = padL + plotW;
  const y1 = padT + plotH;

  function dataToPx(fpr, tpr) {
    return {
      px: x0 + fpr * plotW,
      py: y1 - tpr * plotH,
    };
  }

  addLine(frame, x0 - 4, y0, x0 - 4, y1 + 4, { color: PALETTE.ink, weight: 1 });
  addLine(frame, x0 - 4, y1 + 4, x1, y1 + 4, { color: PALETTE.ink, weight: 1 });
  drawXAxisTicks(frame, x0, y1 + 4, plotW, xb, [0, 0.25, 0.5, 0.75, 1], monoFont);
  drawYAxisTicks(frame, x0 - 4, y0, plotH, yb, [0, 0.25, 0.5, 0.75, 1], monoFont);

  const diag0 = dataToPx(0, 0);
  const diag1 = dataToPx(1, 1);
  addLine(frame, diag0.px, diag0.py, diag1.px, diag1.py, {
    color: PALETTE.muted,
    weight: 0.8,
    dash: [4, 3],
    opacity: 0.7,
  });

  for (let c = 0; c < curves.length; c++) {
    const curve = curves[c];
    const fpr = curve.fpr || [];
    const tpr = curve.tpr || [];
    if (fpr.length < 2) continue;
    const color = hexToRgb(curve.color || "#888888");
    let path = "";
    for (let i = 0; i < fpr.length; i++) {
      const pt = dataToPx(fpr[i], tpr[i]);
      path += (i === 0 ? "M " : " L ") + pt.px + " " + pt.py;
    }
    const vec = figma.createVector();
    vec.vectorPaths = [{ windingRule: "NONZERO", data: path }];
    vec.strokes = [{ type: "SOLID", color: color, opacity: 1 }];
    vec.strokeWeight = 1.6;
    vec.fills = [];
    frame.appendChild(vec);
  }

  addText(frame, panel.x_label || "False positive rate", x0, y1 + 22, bodyFont, {
    size: 7,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yl = addText(frame, panel.y_label || "True positive rate", 2, y0 + plotH * 0.3, bodyFont, {
    size: 7,
    color: PALETTE.muted,
    width: 90,
  });
  yl.rotation = -90;

  let legY = y0 + 4;
  for (let c = 0; c < curves.length; c++) {
    const curve = curves[c];
    const color = hexToRgb(curve.color || "#888888");
    const sw = figma.createRectangle();
    sw.resize(12, 2);
    sw.x = x1 - 118;
    sw.y = legY;
    sw.fills = [solidFill(color)];
    sw.strokes = [];
    frame.appendChild(sw);
    const med = curve.median_auc != null ? curve.median_auc.toFixed(2) : "?";
    const emd = curve.representative_emdb_id || "?";
    const leg = (curve.label || curve.predictor || "") + " (med=" + med + ", EMD-" + emd + ")";
    addText(frame, leg, x1 - 102, legY - 3, bodyFont, {
      size: 5.5,
      color: PALETTE.ink,
      width: 100,
    });
    legY += 11;
  }

  return frame;
}

function buildAllFigure(data, bodyFont, monoFont) {
  const wrapper = figma.createFrame();
  wrapper.name = data.figure_title || "placement utility (all)";
  wrapper.layoutMode = "VERTICAL";
  wrapper.primaryAxisSizingMode = "AUTO";
  wrapper.counterAxisSizingMode = "AUTO";
  wrapper.itemSpacing = 20;
  wrapper.paddingLeft = 16;
  wrapper.paddingRight = 16;
  wrapper.paddingTop = 16;
  wrapper.paddingBottom = 16;
  wrapper.fills = [solidFill(PALETTE.panelBg)];
  wrapper.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 8;
  wrapper.appendChild(buildHeadToHeadFigure(data.panels.head_to_head, bodyFont, monoFont));
  wrapper.appendChild(buildRankRecoveryPanel(data.panels.rank_recovery, bodyFont, monoFont));
  if (data.panels.low_q_roc && (data.panels.low_q_roc.curves || []).length) {
    wrapper.appendChild(buildLowQRocPanel(data.panels.low_q_roc, bodyFont, monoFont));
  }
  return wrapper;
}

function buildCombinedFigure(data, bodyFont, monoFont) {
  const h2h = buildHeadToHeadFigure(data.panels.head_to_head, bodyFont, monoFont);
  const rr = buildRankRecoveryPanel(data.panels.rank_recovery, bodyFont, monoFont);

  const wrapper = figma.createFrame();
  wrapper.name = data.figure_title || "placement utility";
  wrapper.layoutMode = "VERTICAL";
  wrapper.primaryAxisSizingMode = "AUTO";
  wrapper.counterAxisSizingMode = "AUTO";
  wrapper.itemSpacing = 20;
  wrapper.paddingLeft = 16;
  wrapper.paddingRight = 16;
  wrapper.paddingTop = 16;
  wrapper.paddingBottom = 16;
  wrapper.fills = [solidFill(PALETTE.panelBg)];
  wrapper.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 8;
  wrapper.appendChild(h2h);
  wrapper.appendChild(rr);
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
    if (msg.mode === "head_to_head") {
      if (!data.panels.head_to_head) throw new Error("Missing head-to-head data.");
      node = buildHeadToHeadFigure(data.panels.head_to_head, bodyFont, monoFont);
    } else if (msg.mode === "rank_recovery") {
      if (!data.panels.rank_recovery) throw new Error("Missing rank recovery data.");
      node = buildRankRecoveryPanel(data.panels.rank_recovery, bodyFont, monoFont);
    } else if (msg.mode === "both") {
      node = buildCombinedFigure(data, bodyFont, monoFont);
    } else if (msg.mode === "low_q_roc") {
      if (!data.panels.low_q_roc) throw new Error("Missing ROC data.");
      node = buildLowQRocPanel(data.panels.low_q_roc, bodyFont, monoFont);
    } else if (msg.mode === "all") {
      node = buildAllFigure(data, bodyFont, monoFont);
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
