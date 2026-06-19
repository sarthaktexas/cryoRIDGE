/**
 * Thesis Q vs V by Class — Figma plugin
 *
 * Recreates cohort_q_vs_v_by_class.png: box-and-strip summary of
 * ρ(Q-score, constraint V) by coarse protein class.
 */

figma.showUI(__html__, { width: 300, height: 360 });

function hexToRgb(hex) {
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
  white: { r: 1, g: 1, b: 1 },
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
  if (opts.width) {
    node.textAutoResize = "HEIGHT";
    node.resize(opts.width, 20);
  }
  parent.appendChild(node);
  return node;
}

function drawYAxisTicks(frame, xBase, y0, plotH, yMin, yMax, ticks, monoFont) {
  const tickLen = 4;
  const y1 = y0 + plotH;
  for (let i = 0; i < ticks.length; i++) {
    const t = ticks[i];
    const ty = (t - yMin) / (yMax - yMin);
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

function valueToPy(val, y0, plotH, yMin, yMax) {
  const ty = (val - yMin) / (yMax - yMin);
  return y0 + plotH - ty * plotH;
}

function jitterToPx(jitter, groupSpacing, boxWidthFrac) {
  return jitter * (groupSpacing * boxWidthFrac);
}

/**
 * Box-and-strip panel grouped by protein class.
 */
function buildReliabilityByClassPanel(data, bodyFont, monoFont) {
  const panel = data.panel || {};
  const groups = panel.groups || [];
  if (!groups.length) throw new Error("No protein-class groups in export.");

  const yLim = data.y_lim || [-0.05, 1.02];
  const yMin = yLim[0];
  const yMax = yLim[1];
  const boxWidthFrac = panel.box_width != null ? panel.box_width : 0.55;

  const nGroups = groups.length;
  const W = Math.max(480, Math.round(68 * nGroups + 180));
  const H = 360;
  const padL = 56;
  const padR = 16;
  const padT = 42;
  const padB = 88;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const groupSpacing = plotW / nGroups;
  const boxW = groupSpacing * boxWidthFrac;

  const frame = figma.createFrame();
  frame.name = "Q-score vs V by protein class";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [solidFill(PALETTE.panelBg)];
  frame.strokes = [{ type: "SOLID", color: PALETTE.grid }];
  frame.strokeWeight = 1;
  frame.cornerRadius = 6;
  frame.resize(W, H);

  const title = panel.title || data.figure_title || "Q-score vs V by protein class";
  addText(frame, title, 10, 8, bodyFont, { size: 9.5, color: PALETTE.ink, width: W - 20 });

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

  const yTicks = pickTicks(yMin, yMax, 6);
  drawYAxisTicks(frame, x0, y0, plotH, yMin, yMax, yTicks, monoFont);

  const cohortMedian = panel.cohort_median != null ? panel.cohort_median : data.cohort_median;
  if (isFinite(cohortMedian) && cohortMedian >= yMin && cohortMedian <= yMax) {
    const medY = valueToPy(cohortMedian, y0, plotH, yMin, yMax);
    addLine(frame, x0, medY, x1, medY, {
      color: PALETTE.muted,
      weight: 0.7,
      dash: [3, 3],
      opacity: 0.85,
    });
    const medLabel = panel.cohort_median_label || ("cohort median (" + cohortMedian.toFixed(2) + ")");
    addText(frame, medLabel, x1 - 108, medY - 14, bodyFont, {
      size: 6,
      color: PALETTE.muted,
      width: 104,
      align: "RIGHT",
    });
  }

  const ptR = 4.5;

  for (let gi = 0; gi < groups.length; gi++) {
    const grp = groups[gi];
    const pos = grp.position != null ? grp.position : gi + 1;
    const cx = x0 + (pos - 0.5) * groupSpacing;
    const color = hexToRgb(grp.color || "#4B6FD4");
    const box = grp.box || {};

    if (box.whisker_low != null && box.q1 != null) {
      const wLo = valueToPy(box.whisker_low, y0, plotH, yMin, yMax);
      const wQ1 = valueToPy(box.q1, y0, plotH, yMin, yMax);
      addLine(frame, cx, wLo, cx, wQ1, { color: color, weight: 0.6, opacity: 0.9 });
      addLine(frame, cx - boxW * 0.22, wLo, cx + boxW * 0.22, wLo, { color: color, weight: 0.6, opacity: 0.9 });
    }

    if (box.q3 != null && box.whisker_high != null) {
      const wQ3 = valueToPy(box.q3, y0, plotH, yMin, yMax);
      const wHi = valueToPy(box.whisker_high, y0, plotH, yMin, yMax);
      addLine(frame, cx, wQ3, cx, wHi, { color: color, weight: 0.6, opacity: 0.9 });
      addLine(frame, cx - boxW * 0.22, wHi, cx + boxW * 0.22, wHi, { color: color, weight: 0.6, opacity: 0.9 });
    }

    if (box.q1 != null && box.q3 != null) {
      const topY = valueToPy(box.q3, y0, plotH, yMin, yMax);
      const botY = valueToPy(box.q1, y0, plotH, yMin, yMax);
      const rect = figma.createRectangle();
      rect.resize(boxW, Math.max(1, botY - topY));
      rect.x = cx - boxW / 2;
      rect.y = topY;
      rect.fills = [solidFill(color, 0.35)];
      rect.strokes = [{ type: "SOLID", color: color, opacity: 0.95 }];
      rect.strokeWeight = 0.6;
      frame.appendChild(rect);
    }

    if (box.median != null) {
      const medY = valueToPy(box.median, y0, plotH, yMin, yMax);
      addLine(frame, cx - boxW / 2, medY, cx + boxW / 2, medY, {
        color: PALETTE.ink,
        weight: 1,
      });
    }

    const points = grp.points || [];
    for (let pi = 0; pi < points.length; pi++) {
      const pt = points[pi];
      const px = cx + jitterToPx(pt.jitter || 0, groupSpacing, boxWidthFrac);
      const py = valueToPy(pt.value, y0, plotH, yMin, yMax);
      const dot = figma.createEllipse();
      dot.resize(ptR * 2, ptR * 2);
      dot.x = px - ptR;
      dot.y = py - ptR;
      dot.fills = [solidFill(color, 0.9)];
      dot.strokes = [{ type: "SOLID", color: PALETTE.white, opacity: 0.85 }];
      dot.strokeWeight = 0.35;
      frame.appendChild(dot);
    }

    const labelNode = addText(frame, grp.label || "", cx - 36, y1 + 6, bodyFont, {
      size: 6.5,
      color: PALETTE.muted,
      width: 72,
      align: "CENTER",
      rotation: -30 * (Math.PI / 180),
    });
    labelNode.x = cx - 18;
    labelNode.y = y1 + 4;
  }

  const yl = addText(frame, panel.y_label || "", 4, y0 + plotH * 0.35, bodyFont, {
    size: 7.5,
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
        "Missing cohort export. Run: uv run python scripts/run_reliability_by_class_figma_export.py"
      );
    }

    const node = buildReliabilityByClassPanel(data, bodyFont, monoFont);
    figma.currentPage.appendChild(node);
    figma.viewport.scrollAndZoomIntoView([node]);
    figma.ui.postMessage({
      type: "result",
      text: "Created Q vs V by-class figure (" + (data.n_structures || data.n_maps || "?") + " structures).",
    });
  } catch (err) {
    figma.ui.postMessage({ type: "error", text: String(err.message || err) });
  }
};
