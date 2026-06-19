/**
 * Thesis Conformation Pair — Figma plugin
 *
 * Panel B: Cα RMSD vs Δreliability scatter (domain-colored residues).
 */

figma.showUI(__html__, { width: 300, height: 380 });

function hexToRgb(hex) {
  if (!hex || hex.charAt(0) !== "#") return { r: 0.67, g: 0.67, b: 0.67 };
  return {
    r: parseInt(hex.slice(1, 3), 16) / 255,
    g: parseInt(hex.slice(3, 5), 16) / 255,
    b: parseInt(hex.slice(5, 7), 16) / 255,
  };
}

const PALETTE = {
  ink: { r: 0.12, g: 0.12, b: 0.14 },
  muted: { r: 0.42, g: 0.42, b: 0.46 },
  grid: { r: 0.88, g: 0.88, b: 0.9 },
  panelBg: { r: 1, g: 1, b: 1 },
  zeroLine: { r: 0.6, g: 0.6, b: 0.6 },
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
  if (opts.rotation) node.rotation = opts.rotation;
  if (opts.lineHeight) node.lineHeight = { unit: "PIXELS", value: opts.lineHeight };
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
    addText(frame, formatTick(t), xBase - tickLen - 24, py - 4, monoFont, {
      size: 6.5,
      color: PALETTE.muted,
      width: 22,
      align: "RIGHT",
    });
  }
}

function findClearOrigin() {
  let maxX = 0;
  for (let i = 0; i < figma.currentPage.children.length; i++) {
    const node = figma.currentPage.children[i];
    if ("x" in node && "width" in node) {
      const right = node.x + node.width;
      if (right > maxX) maxX = right;
    }
  }
  return { x: maxX + 40, y: 40 };
}

function fmtRho(rho) {
  if (rho == null || !isFinite(rho)) return "n/a";
  return (rho >= 0 ? "+" : "") + rho.toFixed(2);
}

/**
 * Panel B: Cα RMSD vs Δreliability scatter with domain legend.
 */
function buildRmsdDeltaRelPanel(data, bodyFont, monoFont) {
  const panel = data.panel || {};
  const points = panel.points || [];
  if (!points.length) throw new Error("No scatter points in export.");

  const W = 380;
  const H = 340;
  const padL = 56;
  const padR = panel.legend && panel.legend.length ? 88 : 16;
  const padT = 52;
  const padB = 48;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const xs = points.map(function (p) { return p.x; });
  const ys = points.map(function (p) { return p.y; });
  const xb = axisBounds(xs);
  const yb = axisBounds(ys);

  const frame = figma.createFrame();
  frame.name = "panel b: Cα RMSD vs Δreliability";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  const letter = panel.letter || "b";
  addText(frame, letter, 8, 6, bodyFont, { size: 12, color: PALETTE.muted });

  const title = panel.title || "Cα RMSD vs Δreliability";
  addText(frame, title, 24, 8, bodyFont, { size: 9, color: PALETTE.ink, width: W - 32 });

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
  drawXAxisTicks(frame, x0, y1, plotW, xb, xTicks, monoFont);
  drawYAxisTicks(frame, x0, y0, plotH, yb, yTicks, monoFont);

  if (panel.zero_line !== false && yb.min <= 0 && yb.max >= 0) {
    const zy = dataToPx(xb.min, 0).py;
    addLine(frame, x0, zy, x1, zy, { color: PALETTE.zeroLine, weight: 0.5, opacity: 0.9 });
  }

  const ptR = panel.point_radius != null ? panel.point_radius : 2;
  const ptAlpha = panel.point_alpha != null ? panel.point_alpha : 0.6;

  for (let i = 0; i < points.length; i++) {
    const pt = points[i];
    const px = dataToPx(pt.x, pt.y);
    const dot = figma.createEllipse();
    dot.resize(ptR * 2, ptR * 2);
    dot.x = px.px - ptR;
    dot.y = px.py - ptR;
    dot.fills = [solidFill(hexToRgb(pt.color || "#aaaaaa"), ptAlpha)];
    dot.strokes = [];
    frame.appendChild(dot);
  }

  const statsText = panel.stats_text || (
    "Spearman ρ(RMSD, Δrel) = " + fmtRho(data.spearman_rho) + "\nn = " + (data.n_residues || points.length)
  );
  addText(frame, statsText, x0 + 6, y0 + 4, bodyFont, {
    size: 6.5,
    color: PALETTE.ink,
    width: plotW * 0.55,
    lineHeight: 9,
  });

  const legend = panel.legend || [];
  if (legend.length) {
    let ly = y1 - 8 - legend.length * 14;
    for (let li = 0; li < legend.length; li++) {
      const item = legend[li];
      const swatch = figma.createRectangle();
      swatch.resize(8, 8);
      swatch.x = x1 + 8;
      swatch.y = ly;
      swatch.fills = [solidFill(hexToRgb(item.color))];
      swatch.strokes = [];
      frame.appendChild(swatch);

      let label = item.name || "";
      if (item.rho != null && isFinite(item.rho)) {
        label += " (ρ=" + fmtRho(item.rho) + ")";
      }
      addText(frame, label, x1 + 20, ly - 1, bodyFont, {
        size: 5.5,
        color: PALETTE.ink,
        width: padR - 24,
      });
      ly += 14;
    }
  }

  addText(frame, panel.x_label || "", x0, y1 + 16, bodyFont, {
    size: 7,
    color: PALETTE.muted,
    width: plotW,
    align: "CENTER",
  });
  const yl = addText(frame, panel.y_label || "", 4, y0 + plotH * 0.35, bodyFont, {
    size: 7,
    color: PALETTE.muted,
    width: plotH * 0.55,
  });
  yl.rotation = -90 * (Math.PI / 180);

  return frame;
}

figma.ui.onmessage = async function (msg) {
  try {
    const bodyFont = await loadBodyFont();
    const monoFont = await loadMonoFont();
    const data = msg.exportData;

    if (!data || !data.panel) {
      throw new Error(
        "Missing export. Run: uv run python scripts/run_conformation_pair_figma_export.py"
      );
    }

    const node = buildRmsdDeltaRelPanel(data, bodyFont, monoFont);
    const origin = findClearOrigin();
    node.x = origin.x;
    node.y = origin.y;
    figma.currentPage.appendChild(node);
    figma.viewport.scrollAndZoomIntoView([node]);
    figma.ui.postMessage({
      type: "result",
      text: "Created panel B (" + (data.n_residues || "?") + " residues, ρ=" + fmtRho(data.spearman_rho) + ").",
    });
  } catch (err) {
    figma.ui.postMessage({ type: "error", text: String(err.message || err) });
  }
};
