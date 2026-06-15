/**
 * Thesis Grid Schematic — Figma plugin
 *
 * Generates schematic 2D slices + low-poly 3D icons, derives half-maps,
 * and supports thesis-aligned grid operations.
 */

figma.showUI(__html__, { width: 300, height: 640 });

/** Fixed thesis figure stages (left → right). */
const THESIS_PIPELINE = [
  "source",
  "rho_slice",
  "halfmaps",
  "rho",
  "v",
  "reliability",
  "build_zones",
];

const EPS = 1e-6;
const DEFAULT_GRID_SIZE = 8;
const DEFAULT_CELL_PX = 18;
const DEFAULT_CELL_GAP = 0;
const GRID_STROKE_WEIGHT = 0.5;
const GRID_STROKE_COLOR = { r: 0.48, g: 0.48, b: 0.48 };
const GRAY_COLOR = { r: 0.42, g: 0.42, b: 0.42 };

/** User palette — thesis figure colors. */
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
  white: { r: 1, g: 1, b: 1 },
  black: { r: 0, g: 0, b: 0 },
};

const HALFMAP_TINT_STRENGTH = 0.38;
const FONT_CANDIDATES = [
  { family: "Geist Mono", style: "Regular" },
  { family: "Geist Mono", style: "Medium" },
  { family: "Roboto Mono", style: "Regular" },
  { family: "Inter", style: "Regular" },
];

/** @typedef {{ r: number, g: number, b: number }} RGB */
/** @typedef {{ family: string, style: string }} FontName */

/** @type {FontName | null} */
let cachedLabelFont = null;

// ─── PRNG & math ─────────────────────────────────────────────────────────────

/** @param {number} seed */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * @param {number} r
 * @param {number} c
 * @param {number} cr
 * @param {number} cc
 * @param {number} sigma
 * @param {number} amp
 */
function gaussian2(r, c, cr, cc, sigma, amp) {
  const dr = r - cr;
  const dc = c - cc;
  return amp * Math.exp(-(dr * dr + dc * dc) / (2 * sigma * sigma));
}

/**
 * @param {number[][]} grid
 * @param {boolean} stretch
 * @returns {number[][]}
 */
function normalizeGrid01(grid, stretch) {
  let min = Infinity;
  let max = -Infinity;
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      const v = grid[r][c];
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const span = max - min;
  if (span < EPS) return grid.map(function (row) { return row.map(function () { return 0; }); });
  if (!stretch) {
    return grid.map(function (row) {
      return row.map(function (v) { return Math.max(0, Math.min(1, (v - min) / span)); });
    });
  }
  return grid.map(function (row) {
    return row.map(function (v) { return (v - min) / span; });
  });
}

/**
 * @param {number[][]} grid
 * @param {number} passes
 */
function boxSmooth(grid, passes) {
  const rows = grid.length;
  const cols = grid[0].length;
  let cur = grid;
  for (let p = 0; p < passes; p++) {
    /** @type {number[][]} */
    const next = [];
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) {
        let sum = 0;
        let n = 0;
        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const rr = r + dr;
            const cc = c + dc;
            if (rr >= 0 && rr < rows && cc >= 0 && cc < cols) {
              sum += cur[rr][cc];
              n++;
            }
          }
        }
        row.push(sum / n);
      }
      next.push(row);
    }
    cur = next;
  }
  return cur;
}

/**
 * Bilinear sample at fractional (row, col) on a 2D grid.
 * @param {number[][]} grid
 * @param {number} ry
 * @param {number} cx
 */
function bilinearSampleGrid(grid, ry, cx) {
  const rows = grid.length;
  const cols = grid[0].length;
  const r0 = Math.max(0, Math.min(rows - 1, Math.floor(ry)));
  const c0 = Math.max(0, Math.min(cols - 1, Math.floor(cx)));
  const r1 = Math.min(r0 + 1, rows - 1);
  const c1 = Math.min(c0 + 1, cols - 1);
  const tr = ry - r0;
  const tc = cx - c0;
  const v00 = grid[r0][c0];
  const v01 = grid[r0][c1];
  const v10 = grid[r1][c0];
  const v11 = grid[r1][c1];
  return (
    (1 - tr) * (1 - tc) * v00 +
    (1 - tr) * tc * v01 +
    tr * (1 - tc) * v10 +
    tr * tc * v11
  );
}

/**
 * Upsample a coarse grid for smoother mesh lines (bilinear, scipy-style).
 * @param {number[][]} grid
 * @param {number} factor integer steps between original vertices
 */
function upsampleGrid(grid, factor) {
  const steps = Math.max(1, Math.round(factor));
  if (steps <= 1) return grid;
  const rows = grid.length;
  const cols = grid[0].length;
  const fRows = (rows - 1) * steps + 1;
  const fCols = (cols - 1) * steps + 1;
  /** @type {number[][]} */
  const out = [];
  for (let fr = 0; fr < fRows; fr++) {
    const row = [];
    const ry = fr / steps;
    for (let fc = 0; fc < fCols; fc++) {
      row.push(bilinearSampleGrid(grid, ry, fc / steps));
    }
    out.push(row);
  }
  return out;
}

/**
 * Smooth + normalize a grid for mesh topology (0 = base, 1 = peak).
 * @param {number[][]} values
 * @param {{ subdivisions?: number, smoothPasses?: number }} opts
 */
function prepareMeshGrid(values, opts) {
  opts = opts || {};
  const subdivisions = opts.subdivisions != null ? opts.subdivisions : 4;
  const smoothPasses = opts.smoothPasses != null ? opts.smoothPasses : 2;
  let grid = upsampleGrid(values, subdivisions);
  grid = boxSmooth(grid, smoothPasses);
  return normalizeGrid01(grid, true);
}

const PROTEIN_ARCHETYPES = ["tetramer", "kcsa", "gpcr"];
const ARCHETYPE_LABELS = {
  tetramer: "Tetramer",
  kcsa: "Ion channel",
  gpcr: "Membrane protein",
  ion_channel: "Ion channel",
  membrane_protein: "Membrane protein",
};
const MAX_ISO_COLUMNS = 40;

/**
 * Simple 2D slice shapes (schematic only — not structurally accurate).
 * @param {number} size
 * @param {string} archetype
 * @param {() => number} rand
 */
function generateSliceGrid(size, archetype, rand) {
  const cx = (size - 1) * 0.5;
  const cy = (size - 1) * 0.5;
  const s = size * 0.18;
  /** @type {number[][]} */
  const grid = [];

  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) {
      let v = 0;
      if (archetype === "tetramer") {
        const ring = size * 0.22;
        for (let i = 0; i < 4; i++) {
          const ang = i * Math.PI * 0.5 + rand() * 0.2;
          v += gaussian2(r, c, cx + ring * Math.cos(ang), cy + ring * Math.sin(ang), s, 1.0);
        }
        v += gaussian2(r, c, cx, cy, s * 0.55, 0.35);
      } else if (archetype === "kcsa") {
        const ring = size * 0.2;
        for (let i = 0; i < 4; i++) {
          const ang = i * Math.PI * 0.5;
          v += gaussian2(r, c, cx + ring * Math.cos(ang), cy + ring * Math.sin(ang), s * 0.65, 0.95);
        }
        const dr = r - cx;
        const dc = c - cy;
        const dist = Math.sqrt(dr * dr + dc * dc);
        if (dist < size * 0.1) v *= 0.15;
      } else {
        v += gaussian2(r, c, cx, cy, size * 0.28, 0.55);
        v += gaussian2(r, c, cx + size * 0.06, cy - size * 0.08, s * 1.1, 0.85);
        for (let i = 0; i < 5; i++) {
          const ang = i * 1.25;
          v += gaussian2(
            r, c,
            cx + size * 0.12 * Math.cos(ang),
            cy + size * 0.1 * Math.sin(ang),
            s * 0.45, 0.35
          );
        }
      }
      row.push(Math.max(0, v));
    }
    grid.push(row);
  }
  return normalizeGrid01(grid, true);
}

/**
 * @param {string} archetype
 */
function normalizeArchetypeId(archetype) {
  if (archetype === "ion_channel") return "kcsa";
  if (archetype === "membrane_protein") return "gpcr";
  return archetype;
}

/**
 * @param {string} archetype
 * @param {number} seed
 */
function resolveArchetype(archetype, seed) {
  const id = normalizeArchetypeId(archetype);
  if (id && PROTEIN_ARCHETYPES.indexOf(id) >= 0) return id;
  return PROTEIN_ARCHETYPES[seed % PROTEIN_ARCHETYPES.length];
}

/**
 * @param {SceneNode} rect
 * @param {number} [strokeWeight]
 */
function styleGridCell(rect, strokeWeight) {
  rect.strokes = [{ type: "SOLID", color: GRID_STROKE_COLOR, opacity: 1 }];
  rect.strokeWeight = strokeWeight !== undefined ? strokeWeight : GRID_STROKE_WEIGHT;
  rect.strokeAlign = "CENTER";
  rect.cornerRadius = 0;
}

/**
 * Map normalized value 0–1 to dark (low) → light (high).
 * @param {number} t
 */
function grayscaleCellColor(t) {
  const u = Math.max(0, Math.min(1, t));
  return { r: u, g: u, b: u };
}

/**
 * Sample a coarse isometric column field from a 2D density slice.
 * @param {number[][]} grid
 * @param {number} wx
 * @param {number} wy
 * @param {number} maxH
 * @param {number} threshold
 * @param {number} step
 * @param {number} rowOff
 * @param {number} colOff
 */
function collectIsoColumnsFromGrid(grid, wx, wy, maxH, threshold, step, rowOff, colOff) {
  const rows = grid.length;
  const cols = grid[0].length;
  /** @type {Array<{ cx: number, cy: number, h: number, v: number, sortKey: number }>} */
  const columns = [];
  rowOff = rowOff || 0;
  colOff = colOff || 0;

  for (let r = rowOff; r < rows; r += step) {
    for (let c = colOff; c < cols; c += step) {
      let v = 0;
      let n = 0;
      for (let dr = 0; dr < step && r + dr < rows; dr++) {
        for (let dc = 0; dc < step && c + dc < cols; dc++) {
          v += grid[r + dr][c + dc];
          n++;
        }
      }
      v /= n;
      if (v < threshold) continue;
      const cx = (c - r) * wx;
      const cy = (c + r) * wy;
      columns.push({
        cx: cx,
        cy: cy,
        h: v * maxH,
        v: v,
        sortKey: r + c,
      });
    }
  }
  return columns;
}

/**
 * @param {Array<{ cx: number, cy: number, h: number, v: number }>} columns
 * @param {number} wx
 * @param {number} wy
 */
function mergeIsoColumns(columns, wx, wy) {
  /** @type {Array<{ cx: number, cy: number, h: number, v: number, sortKey: number }>} */
  const merged = [];
  const minDist = wx * 0.85;
  for (let i = 0; i < columns.length; i++) {
    const col = columns[i];
    let tooClose = false;
    for (let j = 0; j < merged.length; j++) {
      const dx = col.cx - merged[j].cx;
      const dy = col.cy - merged[j].cy;
      if (dx * dx + dy * dy < minDist * minDist) {
        if (col.v > merged[j].v) merged[j] = col;
        tooClose = true;
        break;
      }
    }
    if (!tooClose) merged.push(col);
  }
  merged.sort(function (a, b) {
    return a.sortKey - b.sortKey || a.cy - b.cy;
  });
  if (merged.length > MAX_ISO_COLUMNS) {
    merged.sort(function (a, b) { return b.v - a.v; });
    merged.length = MAX_ISO_COLUMNS;
    merged.sort(function (a, b) {
      return a.sortKey - b.sortKey || a.cy - b.cy;
    });
  }
  return merged;
}

/**
 * @param {Array<{ cx: number, cy: number, h: number }>} columns
 * @param {number} wx
 * @param {number} wy
 */
function isoColumnBounds(columns, wx, wy) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < columns.length; i++) {
    const col = columns[i];
    const top = col.cy - col.h;
    minX = Math.min(minX, col.cx - wx);
    maxX = Math.max(maxX, col.cx + wx);
    minY = Math.min(minY, top - wy);
    maxY = Math.max(maxY, col.cy + wy);
  }
  if (!Number.isFinite(minX)) {
    return { minX: -40, minY: -40, maxX: 40, maxY: 40 };
  }
  return { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
}

/**
 * @param {Array<[number, number]>} pts
 * @param {number} dx
 * @param {number} dy
 */
function shiftIsoPts(pts, dx, dy) {
  const out = [];
  for (let i = 0; i < pts.length; i++) {
    out.push([pts[i][0] + dx, pts[i][1] + dy]);
  }
  return out;
}

/**
 * @param {Array<{ pts: Array<[number, number]>, color: RGB, opacity: number }>} faces
 * @param {number} cx
 * @param {number} cy
 * @param {number} h
 * @param {number} wx
 * @param {number} wy
 * @param {RGB} topColor
 * @param {RGB} leftColor
 * @param {RGB} rightColor
 * @param {number} v
 */
function pushIsoColumnFaces(faces, cx, cy, h, wx, wy, topColor, leftColor, rightColor, v) {
  const top = cy - h;
  const op = 0.38 + v * 0.58;
  faces.push({
    pts: [[cx + wx, cy], [cx + wx, top], [cx, top + wy], [cx, cy + wy]],
    color: rightColor,
    opacity: op,
  });
  faces.push({
    pts: [[cx - wx, cy], [cx, cy + wy], [cx, top + wy], [cx - wx, top]],
    color: leftColor,
    opacity: op,
  });
  faces.push({
    pts: [[cx, top - wy], [cx + wx, top], [cx, top + wy], [cx - wx, top]],
    color: topColor,
    opacity: Math.min(1, op + 0.07),
  });
}

/**
 * Isometric 3D icon derived from the full density slice.
 * @param {string} archetype
 * @param {RGB} baseColor
 * @param {number[][]} fullSlice
 */
function buildProteinSilhouette(archetype, baseColor, fullSlice) {
  const frame = figma.createFrame();
  frame.name = "protein 3D";
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [];

  const wx = 6.5;
  const wy = 3.25;
  const maxH = 36;
  const threshold = 0.07;
  const step = fullSlice && fullSlice.length >= 12 ? 2 : 1;
  const topColor = shadeColor(baseColor, 1.15);
  const leftColor = shadeColor(baseColor, 0.8);
  const rightColor = shadeColor(baseColor, 0.55);

  /** @type {Array<{ cx: number, cy: number, h: number, v: number, sortKey: number }>} */
  let columns = [];

  if (fullSlice && fullSlice.length > 0) {
    columns = collectIsoColumnsFromGrid(fullSlice, wx, wy, maxH, threshold, step, 0, 0);
    if (columns.length < MAX_ISO_COLUMNS - 6) {
      const stagger = collectIsoColumnsFromGrid(fullSlice, wx, wy, maxH, threshold + 0.04, step, 1, 1);
      for (let si = 0; si < stagger.length; si++) {
        columns.push(stagger[si]);
      }
    }
    columns = mergeIsoColumns(columns, wx, wy);
  }

  if (columns.length === 0) {
    const fallback = [
      { cx: -20, cy: -6, h: 0.85 * maxH, v: 0.85, sortKey: 0 },
      { cx: 20, cy: -6, h: 0.82 * maxH, v: 0.82, sortKey: 1 },
      { cx: -20, cy: 16, h: 0.78 * maxH, v: 0.78, sortKey: 2 },
      { cx: 20, cy: 16, h: 0.76 * maxH, v: 0.76, sortKey: 3 },
      { cx: 0, cy: 5, h: 0.55 * maxH, v: 0.55, sortKey: 4 },
    ];
    columns = fallback;
  }

  const bounds = isoColumnBounds(columns, wx, wy);
  const pad = 14;
  const dx = pad - bounds.minX;
  const dy = pad - bounds.minY;
  const localW = bounds.maxX - bounds.minX;
  const localH = bounds.maxY - bounds.minY;

  /** @type {Array<{ pts: Array<[number, number]>, color: RGB, opacity: number }>} */
  const faces = [];
  for (let i = 0; i < columns.length; i++) {
    const col = columns[i];
    pushIsoColumnFaces(
      faces, col.cx, col.cy, col.h, wx, wy,
      topColor, leftColor, rightColor, col.v
    );
  }

  for (let fi = 0; fi < faces.length; fi++) {
    addIsoFace(
      frame,
      shiftIsoPts(faces[fi].pts, dx, dy),
      faces[fi].color,
      faces[fi].opacity
    );
  }

  frame.resize(
    Math.max(localW + pad * 2, 72),
    Math.max(localH + pad * 2, 72)
  );
  return frame;
}

/**
 * @param {number[][]} grid
 * @param {number} r0
 * @param {number} c0
 * @param {number} rows
 * @param {number} cols
 */
function cropGrid(grid, r0, c0, rows, cols) {
  /** @type {number[][]} */
  const out = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      row.push(grid[r0 + r][c0 + c]);
    }
    out.push(row);
  }
  return out;
}

/**
 * @param {number} size
 * @param {number} r
 * @param {number} c
 */
function flexibleMask(size, r, c) {
  const fr = (r - size * 0.52) / (size * 0.35);
  const fc = (c - size * 0.48) / (size * 0.35);
  return Math.max(0, Math.min(1, fr * 0.6 + fc * 0.4));
}

/**
 * @param {number[][]} base
 * @param {number} seed
 * @param {number} disagreement
 */
function generateHalfMapsFromBase(base, seed, disagreement) {
  const rand = mulberry32(seed);
  const size = base.length;
  const noiseScale = 0.06 + disagreement * 0.12;

  /** @type {number[][]} */
  const h1 = [];
  /** @type {number[][]} */
  const h2 = [];

  for (let r = 0; r < size; r++) {
    const row1 = [];
    const row2 = [];
    for (let c = 0; c < size; c++) {
      const b = base[r][c];
      const flex = flexibleMask(size, r, c);
      const localDis = disagreement * (0.25 + 0.75 * flex);
      const n1 = (rand() - 0.5) * noiseScale;
      const n2 = (rand() - 0.5) * noiseScale;
      const split = (rand() - 0.5) * localDis * Math.max(b, 0.12);
      row1.push(Math.max(0, Math.min(1, b + n1 + split * 0.5)));
      row2.push(Math.max(0, Math.min(1, b + n2 - split * 0.5)));
    }
    h1.push(row1);
    h2.push(row2);
  }

  return { h1: boxSmooth(h1, 1), h2: boxSmooth(h2, 1) };
}

/**
 * @param {number} size
 * @param {number} seed
 * @param {number} disagreement
 * @param {string} archetypeChoice
 */
function generateProteinSceneData(size, seed, disagreement, archetypeChoice) {
  const rand = mulberry32(seed);
  const fullSize = size + 8;
  const archetype = resolveArchetype(archetypeChoice, seed);
  const fullSlice = generateSliceGrid(fullSize, archetype, rand);
  const off = Math.floor((fullSize - size) / 2);
  const slice = cropGrid(fullSlice, off, off, size, size);
  const maps = generateHalfMapsFromBase(slice, seed + 101, disagreement);
  return {
    fullSlice: fullSlice,
    slice: slice,
    h1: maps.h1,
    h2: maps.h2,
    fullSize: fullSize,
    cropOffset: off,
    archetype: archetype,
    archetypeLabel: ARCHETYPE_LABELS[archetype] || archetype,
  };
}

// ─── Fonts ───────────────────────────────────────────────────────────────────

async function loadLabelFont() {
  if (cachedLabelFont) return cachedLabelFont;
  for (let i = 0; i < FONT_CANDIDATES.length; i++) {
    try {
      await figma.loadFontAsync(FONT_CANDIDATES[i]);
      cachedLabelFont = FONT_CANDIDATES[i];
      return cachedLabelFont;
    } catch (e) {
      // try next
    }
  }
  throw new Error("Could not load Geist Mono. Install the font in Figma or the OS.");
}

/**
 * @param {number} v
 * @param {"percent" | "zone"} kind
 */
function formatDisplayNumber(v, kind) {
  if (!isFinite(v)) return "0";
  kind = kind || "percent";

  if (kind === "percent") {
    const n = Math.round(Math.max(0, Math.min(1, v)) * 100);
    return String(Math.min(99, n));
  }

  if (kind === "zone") {
    const z = Math.round(v);
    if (z <= 0) return "Om";
    if (z === 1) return "Ct";
    return "Bd";
  }

  return formatDisplayNumber(v, "percent");
}

/**
 * @param {number} v
 */
function formatCellNumber(v) {
  return formatDisplayNumber(v, "percent");
}

/** Build-zone colors: omit = red, caution = amber, build = green. */
const BUILD_ZONE_COLORS = {
  0: PALETTE.red,
  1: PALETTE.amber,
  2: PALETTE.green,
};

/**
 * @param {number} t 0–1
 * @param {RGB} tint
 * @param {number} strength
 */
function tintedGrayscaleCellColor(t, tint, strength) {
  return lerpColor(grayscaleCellColor(t), tint, strength);
}

/**
 * @param {number[][]} grid
 * @param {boolean[][]} mask
 */
function normalizeInMask(grid, mask) {
  let min = Infinity;
  let max = -Infinity;
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      if (mask[r][c]) {
        const v = grid[r][c];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
  }
  const span = max - min;
  if (span < EPS) {
    return grid.map(function (row) { return row.map(function () { return 0; }); });
  }
  return grid.map(function (row, r) {
    return row.map(function (v, c) {
      if (!mask[r][c]) return 0;
      return (v - min) / span;
    });
  });
}

/**
 * @param {number[]} sorted
 * @param {number} pct 0–100
 */
function percentileValue(sorted, pct) {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const idx = (sorted.length - 1) * pct / 100;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/**
 * Classify reliability score into build zones (0=omit, 1=caution, 2=build).
 * @param {number[][]} reliability
 * @param {boolean[][]} mask
 */
function classifyBuildZones(reliability, mask) {
  /** @type {number[]} */
  const vals = [];
  for (let r = 0; r < reliability.length; r++) {
    for (let c = 0; c < reliability[0].length; c++) {
      if (mask[r][c]) vals.push(reliability[r][c]);
    }
  }
  vals.sort(function (a, b) { return a - b; });
  const tLo = percentileValue(vals, 33.33);
  const tHi = percentileValue(vals, 66.67);
  return reliability.map(function (row, r) {
    return row.map(function (v, c) {
      if (!mask[r][c]) return 0;
      if (v < tLo) return 0;
      if (v < tHi) return 1;
      return 2;
    });
  });
}

/**
 * @param {number} t 0–1
 */
function sequentialVColor(t) {
  const u = Math.max(0, Math.min(1, t));
  return lerpColor({ r: 0.88, g: 0.97, b: 0.98 }, PALETTE.cyan, u);
}

/**
 * @param {RGB} c1
 * @param {RGB} c2
 * @param {number} t
 */
function lerpColor(c1, c2, t) {
  const u = Math.max(0, Math.min(1, t));
  return {
    r: c1.r + (c2.r - c1.r) * u,
    g: c1.g + (c2.g - c1.g) * u,
    b: c1.b + (c2.b - c1.b) * u,
  };
}

/**
 * @param {number[][]} grid
 * @param {number} threshold
 */
function maskFromGrid(grid, threshold) {
  return grid.map(function (row) {
    return row.map(function (v) { return v >= threshold; });
  });
}

/**
 * @param {number[][]} grid
 * @param {boolean[][]} mask
 */
function percentileRankInMask(grid, mask) {
  /** @type {Array<{ r: number, c: number, v: number }>} */
  const pairs = [];
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[0].length; c++) {
      if (mask[r][c]) pairs.push({ r: r, c: c, v: grid[r][c] });
    }
  }
  pairs.sort(function (a, b) { return a.v - b.v; });
  /** @type {number[][]} */
  const out = grid.map(function (row) {
    return row.map(function () { return 0; });
  });
  const n = pairs.length;
  for (let i = 0; i < n; i++) {
    out[pairs[i].r][pairs[i].c] = n > 0 ? (i + 1) / n : 0;
  }
  return out;
}

/**
 * @param {RGB} color
 */
function textColorForBg(color) {
  const lum = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
  return lum < 0.52
    ? { r: 1, g: 1, b: 1 }
    : { r: 0, g: 0, b: 0 };
}

/**
 * @param {{ h1: number[][], h2: number[][], slice: number[][] }} data
 */
function computePipelineMetrics(data) {
  const rho = computeOperation("average", data.h1, data.h2);
  const V = computeOperation("gradient_v", rho, null);
  const mask = maskFromGrid(data.slice, 0.12);
  const reliability = percentileRankInMask(V, mask);
  const Vnorm = normalizeInMask(V, mask);
  const buildZones = classifyBuildZones(reliability, mask);
  return {
    rho: rho,
    Vnorm: Vnorm,
    reliability: reliability,
    buildZones: buildZones,
    mask: mask,
  };
}

// ─── Grid I/O (selection → topology) ─────────────────────────────────────────

function isCellRect(node) {
  if (!("fills" in node) || node.type === "TEXT") return false;
  if (!("width" in node) || !("height" in node)) return false;
  return node.width > 0 && node.height > 0;
}

function luminanceFromFill(node) {
  const fills = node.fills;
  if (fills === figma.mixed || !Array.isArray(fills) || fills.length === 0) return 0;
  for (let i = 0; i < fills.length; i++) {
    const fill = fills[i];
    if (fill.type !== "SOLID" || fill.visible === false) continue;
    if (fill.color) {
      const lum = (fill.color.r + fill.color.g + fill.color.b) / 3;
      let opacity = fill.opacity !== undefined ? fill.opacity : 1;
      if ("a" in fill.color && fill.color.a !== undefined) {
        opacity *= fill.color.a;
      }
      if (opacity < 0.05) continue;
      return Math.max(0, Math.min(1, lum));
    }
  }
  return 0;
}

/**
 * Parse displayed cell text (0–99 percent, decimals, zone codes) to 0–1.
 * @param {string} raw
 */
function parseCellLabel(raw) {
  const s = String(raw).trim();
  if (s === "Om") return 0.1;
  if (s === "Ct") return 0.55;
  if (s === "Bd") return 0.9;
  const n = parseFloat(s);
  if (!isFinite(n)) return null;
  if (n > 1 && n <= 99) return n / 100;
  return Math.max(0, Math.min(1, n));
}

function inferGridShape(n) {
  const side = Math.round(Math.sqrt(n));
  if (side * side === n) return { rows: side, cols: side };
  return { rows: 1, cols: n };
}

/**
 * Pick the inner grid frame when the user selects a stage wrapper.
 * @param {SceneNode} root
 */
function resolveGridFrame(root) {
  /** @type {Array<{ frame: SceneNode, nCells: number }>} */
  const candidates = [];
  /** @type {SceneNode[]} */
  const frames = [root];
  if ("children" in root) {
    for (let i = 0; i < root.children.length; i++) {
      if (root.children[i].type === "FRAME") frames.push(root.children[i]);
    }
  }
  for (let fi = 0; fi < frames.length; fi++) {
    const f = frames[fi];
    if (!("findAll" in f)) continue;
    const nCells = f.findAll(function (n) {
      return n.type === "RECTANGLE" && isCellRect(n);
    }).length;
    if (nCells >= 4) candidates.push({ frame: f, nCells: nCells });
  }
  if (candidates.length === 0) return root;
  candidates.sort(function (a, b) { return b.nCells - a.nCells; });
  if (candidates.length > 1 && candidates[0].nCells === candidates[1].nCells) {
    throw new Error("Multiple grids in selection — select one grid frame (e.g. ρ̄ or h₁).");
  }
  return candidates[0].frame;
}

/**
 * Read cell values from a plugin-style grid frame (rect + optional text per cell).
 * @param {SceneNode} root
 */
function gridValuesFromFrame(root) {
  root = resolveGridFrame(root);
  /** @type {Array<GeometryMixin & SceneNode>} */
  const rects = [];
  /** @type {TextNode[]} */
  const texts = [];

  if ("findAll" in root) {
    const all = root.findAll(function (n) {
      return n.type === "RECTANGLE" || n.type === "TEXT";
    });
    for (let i = 0; i < all.length; i++) {
      const n = all[i];
      if (n.type === "TEXT") texts.push(n);
      else if (isCellRect(n)) rects.push(n);
    }
  } else if (isCellRect(root)) {
    rects.push(root);
  }

  if (rects.length < 4) {
    throw new Error('Select a grid frame with at least 4 cells (e.g. "ρ̄", h₁, h₂).');
  }

  rects.sort(function (a, b) {
    const dy = a.y - b.y;
    if (Math.abs(dy) > 0.5) return dy;
    return a.x - b.x;
  });

  /** @type {number[]} */
  const flat = [];
  for (let ri = 0; ri < rects.length; ri++) {
    const rect = rects[ri];
    let v = null;
    for (let ti = 0; ti < texts.length; ti++) {
      const txt = texts[ti];
      if (Math.abs(txt.x - rect.x) < 1.5 && Math.abs(txt.y - rect.y) < 1.5) {
        v = parseCellLabel(txt.characters);
        break;
      }
    }
    flat.push(v != null ? v : luminanceFromFill(rect));
  }

  const shape = inferGridShape(flat.length);
  /** @type {number[][]} */
  const grid = [];
  let i = 0;
  for (let r = 0; r < shape.rows; r++) {
    const row = [];
    for (let c = 0; c < shape.cols; c++) {
      row.push(i < flat.length ? flat[i] : 0);
      i++;
    }
    grid.push(row);
  }
  return grid;
}

function placeBeside(source, node, gap) {
  gap = gap != null ? gap : 12;
  node.x = source.x + source.width + gap;
  node.y = source.y + Math.max(0, (source.height - node.height) / 2);
  if (node.parent !== figma.currentPage) {
    figma.currentPage.appendChild(node);
  }
}

// ─── Operations ──────────────────────────────────────────────────────────────

function gradMagSq(grid, r, c) {
  const rows = grid.length;
  const cols = grid[0].length;
  const rho = grid[r][c];
  const left = c > 0 ? grid[r][c - 1] : rho;
  const right = c < cols - 1 ? grid[r][c + 1] : rho;
  const up = r > 0 ? grid[r - 1][c] : rho;
  const down = r < rows - 1 ? grid[r + 1][c] : rho;
  const drdx = (right - left) * 0.5;
  const drdy = (down - up) * 0.5;
  return drdx * drdx + drdy * drdy;
}

/** Single-grid ops exposed in the UI (selection → derived grid). */
const GRID_OPERATION_SPECS = {
  gradient_v: {
    label: "Constraint V",
    outputName: "V",
    colorMode: "sequential_v",
    needsMask: true,
  },
  reliability: {
    label: "Reliability score",
    outputName: "reliability",
    colorMode: "reliability",
    needsMask: true,
  },
  build_zones: {
    label: "Build zones",
    outputName: "zones",
    colorMode: "build_zone",
    needsMask: true,
  },
  normalize_in_mask: {
    label: "Normalize in mask",
    outputName: "normalized",
    colorMode: "grayscale",
    needsMask: true,
  },
};

function computeOperation(op, g1, g2) {
  const rows = g1.length;
  const cols = g1[0].length;
  /** @type {number[][]} */
  const out = [];

  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const a = g1[r][c];
      const b = g2 ? g2[r][c] : 0;
      let v = 0;
      if (op === "average") {
        v = 0.5 * (a + b);
      } else if (op === "gradient_v") {
        v = 0.5 * gradMagSq(g1, r, c);
      } else {
        throw new Error("Unknown operation: " + op);
      }
      row.push(v);
    }
    out.push(row);
  }
  return out;
}

/**
 * Mask for pipeline ops. Density inputs use threshold; derived maps use v &gt; 0.
 * @param {number[][]} grid
 * @param {string} op
 * @param {number} maskThreshold
 */
function resolveOperationMask(grid, op, maskThreshold) {
  if (op === "gradient_v" || op === "normalize_in_mask") {
    return maskFromGrid(grid, maskThreshold);
  }
  return grid.map(function (row) {
    return row.map(function (v) { return v > EPS; });
  });
}

/**
 * Apply a thesis pipeline op to one input grid.
 * @param {string} op
 * @param {number[][]} grid
 * @param {{ maskThreshold?: number }} opts
 */
function applySingleGridOperation(op, grid, opts) {
  opts = opts || {};
  const spec = GRID_OPERATION_SPECS[op];
  if (!spec) {
    throw new Error("Unknown grid operation: " + op);
  }

  const maskThreshold = opts.maskThreshold != null ? opts.maskThreshold : 0.12;
  const mask = resolveOperationMask(grid, op, maskThreshold);

  if (op === "gradient_v") {
    const raw = computeOperation("gradient_v", grid, null);
    return {
      values: normalizeInMask(raw, mask),
      mask: mask,
      colorMode: spec.colorMode,
      outputName: spec.outputName,
    };
  }

  if (op === "reliability") {
    return {
      values: percentileRankInMask(grid, mask),
      mask: mask,
      colorMode: spec.colorMode,
      outputName: spec.outputName,
    };
  }

  if (op === "build_zones") {
    return {
      values: classifyBuildZones(grid, mask),
      mask: mask,
      colorMode: spec.colorMode,
      outputName: spec.outputName,
    };
  }

  if (op === "normalize_in_mask") {
    return {
      values: normalizeInMask(grid, mask),
      mask: mask,
      colorMode: spec.colorMode,
      outputName: spec.outputName,
    };
  }

  throw new Error("Unhandled grid operation: " + op);
}

/**
 * Match output cell size to a source grid frame.
 * @param {SceneNode} gridFrame
 */
function inferCellPxFromFrame(gridFrame) {
  if (!("findAll" in gridFrame)) return DEFAULT_CELL_PX;
  const rects = gridFrame.findAll(function (n) {
    return n.type === "RECTANGLE" && isCellRect(n);
  });
  if (rects.length < 2) {
    return rects.length === 1 && "width" in rects[0] ? rects[0].width : DEFAULT_CELL_PX;
  }
  rects.sort(function (a, b) {
    const dy = a.y - b.y;
    if (Math.abs(dy) > 0.5) return dy;
    return a.x - b.x;
  });
  const cellW = rects[0].width;
  const sameRow = rects.find(function (r) {
    return r !== rects[0] && Math.abs(r.y - rects[0].y) < 0.5;
  });
  if (sameRow && "x" in sameRow) {
    const gap = sameRow.x - (rects[0].x + rects[0].width);
    if (gap >= 0 && gap < 12) return { cellPx: cellW, cellGap: gap };
  }
  return { cellPx: cellW, cellGap: DEFAULT_CELL_GAP };
}

/**
 * Build a styled output grid beside a selected input grid.
 * @param {SceneNode} source
 * @param {string} op
 * @param {{ maskThreshold?: number, fontName: FontName }} opts
 */
function buildDerivedGridFromSelection(source, op, opts) {
  const gridFrame = resolveGridFrame(source);
  const input = gridValuesFromFrame(source);
  const result = applySingleGridOperation(op, input, {
    maskThreshold: opts.maskThreshold,
  });
  const layout = inferCellPxFromFrame(gridFrame);
  const cellPx = typeof layout === "number" ? layout : layout.cellPx;
  const cellGap = typeof layout === "number" ? DEFAULT_CELL_GAP : layout.cellGap;
  const spec = GRID_OPERATION_SPECS[op];

  let outFrame;
  if (spec.colorMode === "grayscale") {
    outFrame = buildFlatGrid(result.values, {
      label: result.outputName,
      cellPx: cellPx,
      cellGap: cellGap,
      showNumbers: true,
      fontName: opts.fontName,
      colorMode: "grayscale",
    });
  } else {
    outFrame = buildColoredGrid(result.values, {
      label: result.outputName,
      cellPx: cellPx,
      cellGap: cellGap,
      fontName: opts.fontName,
      colorMode: spec.colorMode,
    });
  }

  outFrame.name = spec.outputName + " ← " + (gridFrame.name || "grid");
  placeBeside(gridFrame, outFrame, 12);
  return { gridFrame: gridFrame, outFrame: outFrame, rows: input.length, cols: input[0].length };
}

function solidFill(color, opacity) {
  return {
    type: "SOLID",
    color: { r: color.r, g: color.g, b: color.b },
    opacity: Math.max(0, Math.min(1, opacity)),
  };
}

function shadeColor(color, factor) {
  return {
    r: Math.max(0, Math.min(1, color.r * factor)),
    g: Math.max(0, Math.min(1, color.g * factor)),
    b: Math.max(0, Math.min(1, color.b * factor)),
  };
}

// ─── Building 2D grids ───────────────────────────────────────────────────────

/**
 * @param {number[][]} values
 * @param {{ label: string, baseColor: RGB, opacityLow: number, opacityHigh: number, cellPx: number, cellGap: number, strokeWeight?: number, showNumbers?: boolean, fontName?: FontName, highlight?: { r0: number, c0: number, rows: number, cols: number } | null }} opts
 */
function buildFlatGrid(values, opts) {
  const rows = values.length;
  const cols = values[0].length;
  const cellW = opts.cellPx;
  const cellH = opts.cellPx;
  const gapX = opts.cellGap !== undefined ? opts.cellGap : DEFAULT_CELL_GAP;
  const gapY = gapX;
  const showNumbers = opts.showNumbers === true && opts.fontName;

  const frame = figma.createFrame();
  frame.name = opts.label;
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [];
  frame.resize(cols * cellW + Math.max(0, cols - 1) * gapX, rows * cellH + Math.max(0, rows - 1) * gapY);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const rect = figma.createRectangle();
      rect.name = "cell";
      rect.resize(cellW, cellH);
      rect.x = c * (cellW + gapX);
      rect.y = r * (cellH + gapY);
      const t = values[r][c];
      let fillColor = grayscaleCellColor(t);
      if (opts.colorMode === "halfmap_h1") {
        fillColor = tintedGrayscaleCellColor(t, PALETTE.blue, HALFMAP_TINT_STRENGTH);
      } else if (opts.colorMode === "halfmap_h2") {
        fillColor = tintedGrayscaleCellColor(t, PALETTE.red, HALFMAP_TINT_STRENGTH);
      }
      rect.fills = [solidFill(fillColor, 1)];
      styleGridCell(rect, opts.strokeWeight);
      frame.appendChild(rect);

      if (showNumbers && opts.fontName) {
        const txt = figma.createText();
        txt.fontName = opts.fontName;
        txt.fontSize = Math.max(7, Math.min(11, Math.floor(cellH * 0.42)));
        txt.characters = formatCellNumber(t);
        txt.fills = [{ type: "SOLID", color: textColorForBg(fillColor) }];
        txt.textAlignHorizontal = "CENTER";
        txt.textAlignVertical = "CENTER";
        txt.resize(cellW, cellH);
        txt.x = rect.x;
        txt.y = rect.y;
        frame.appendChild(txt);
      }
    }
  }

  if (opts.highlight) {
    const hl = opts.highlight;
    const box = figma.createRectangle();
    box.name = "slice window";
    box.x = hl.c0 * (cellW + gapX);
    box.y = hl.r0 * (cellH + gapY);
    box.resize(
      hl.cols * cellW + Math.max(0, hl.cols - 1) * gapX,
      hl.rows * cellH + Math.max(0, hl.rows - 1) * gapY
    );
    box.fills = [];
    box.strokes = [{ type: "SOLID", color: PALETTE.red, opacity: 1 }];
    box.strokeWeight = 1.5;
    box.strokeAlign = "CENTER";
    box.dashPattern = [4, 3];
    box.cornerRadius = 0;
    frame.appendChild(box);
  }

  return frame;
}

/**
 * Grid with per-cell fill color and custom labels (locRes, Q-score, reliability).
 * @param {number[][]} values normalized 0-1 for reliability/gray
 * @param {{ label: string, cellPx: number, cellGap: number, fontName: FontName, colorMode: string, rawValues?: number[][], median?: number, spread?: number }} opts
 */
function buildColoredGrid(values, opts) {
  const rows = values.length;
  const cols = values[0].length;
  const cellW = opts.cellPx;
  const cellH = opts.cellPx;
  const gapX = opts.cellGap !== undefined ? opts.cellGap : DEFAULT_CELL_GAP;
  const gapY = gapX;

  const frame = figma.createFrame();
  frame.name = opts.label;
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [];
  frame.resize(cols * cellW + Math.max(0, cols - 1) * gapX, rows * cellH + Math.max(0, rows - 1) * gapY);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const rect = figma.createRectangle();
      rect.name = "cell";
      rect.resize(cellW, cellH);
      rect.x = c * (cellW + gapX);
      rect.y = r * (cellH + gapY);

      let fillColor = GRAY_COLOR;
      let label = formatCellNumber(values[r][c]);
      const t = values[r][c];

      if (opts.colorMode === "reliability") {
        fillColor = lerpColor(PALETTE.white, PALETTE.green, t);
        label = formatDisplayNumber(t, "percent");
      } else if (opts.colorMode === "sequential_v") {
        fillColor = sequentialVColor(t);
        label = formatDisplayNumber(t, "percent");
      } else if (opts.colorMode === "build_zone") {
        const zone = Math.round(values[r][c]);
        fillColor = BUILD_ZONE_COLORS[zone] || BUILD_ZONE_COLORS[0];
        label = formatDisplayNumber(zone, "zone");
      }

      rect.fills = [solidFill(fillColor, 1)];
      styleGridCell(rect, opts.strokeWeight);
      frame.appendChild(rect);

      const txt = figma.createText();
      txt.fontName = opts.fontName;
      txt.fontSize = Math.max(7, Math.min(10, Math.floor(cellH * 0.4)));
      txt.characters = label;
      txt.fills = [{ type: "SOLID", color: textColorForBg(fillColor) }];
      txt.textAlignHorizontal = "CENTER";
      txt.textAlignVertical = "CENTER";
      txt.resize(cellW, cellH);
      txt.x = rect.x;
      txt.y = rect.y;
      frame.appendChild(txt);
    }
  }

  return frame;
}

/**
 * @param {FrameNode} parent
 * @param {number} x0
 * @param {number} y0
 * @param {number} x1
 * @param {number} y1
 */
function addFlowArrow(parent, x0, y0, x1, y1) {
  const shaft = figma.createVector();
  shaft.name = "arrow";
  shaft.vectorPaths = [{
    windingRule: "NONZERO",
    data: "M " + x0 + " " + y0 + " L " + (x1 - 7) + " " + y1,
  }];
  shaft.strokes = [{ type: "SOLID", color: { r: 0.42, g: 0.42, b: 0.46 }, opacity: 1 }];
  shaft.strokeWeight = 1.5;
  parent.appendChild(shaft);

  const head = figma.createVector();
  head.vectorPaths = [{
    windingRule: "NONZERO",
    data: "M " + (x1 - 7) + " " + (y1 - 4) + " L " + x1 + " " + y1 + " L " + (x1 - 7) + " " + (y1 + 4) + " Z",
  }];
  head.fills = [{ type: "SOLID", color: { r: 0.42, g: 0.42, b: 0.46 }, opacity: 1 }];
  head.strokes = [];
  parent.appendChild(head);
}

/**
 * @param {string} title
 * @param {SceneNode[]} nodes
 * @param {FontName} fontName
 * @param {number} pad
 */
function wrapStage(title, nodes, fontName, pad) {
  const group = figma.createFrame();
  group.name = "stage: " + title;
  group.layoutMode = "NONE";
  group.clipsContent = false;
  group.fills = [];

  const cap = createCaption(title, fontName);
  cap.x = pad;
  cap.y = 0;
  group.appendChild(cap);

  let maxW = cap.width + pad * 2;
  let maxH = cap.height + 8;
  const nodeY = cap.height + 10;
  let cx = pad;

  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    n.x = cx;
    n.y = nodeY;
    group.appendChild(n);
    cx += n.width + 10;
    maxW = Math.max(maxW, cx + pad);
    maxH = Math.max(maxH, nodeY + n.height + pad);
  }

  group.resize(maxW, maxH);
  return group;
}

/**
 * Build full thesis pipeline diagram.
 * @param {*} data protein scene data
 * @param {*} metrics pipeline metrics
 * @param {{ seed: number, size: number, include3d: boolean, cellPx: number, cellGap: number, fontName: FontName }} opts
 */
async function generatePipelineDiagram(data, metrics, opts) {
  const origin = findClearOrigin();
  const gridOpts = {
    cellPx: opts.cellPx,
    cellGap: opts.cellGap,
    fontName: opts.fontName,
  };
  const sliceCellPx = Math.max(10, Math.floor(opts.cellPx * 0.72));

  /** @type {Record<string, FrameNode>} */
  const stageFrames = {};

  function makeDensityGrid(values, label, colorMode) {
    return buildFlatGrid(values, {
      label: label,
      cellPx: gridOpts.cellPx,
      cellGap: gridOpts.cellGap,
      showNumbers: true,
      fontName: opts.fontName,
      colorMode: colorMode || "grayscale",
    });
  }

  stageFrames.source = wrapStage(data.archetypeLabel + " — 3D + slice", [
    buildProteinSilhouette(data.archetype, PALETTE.blue, data.fullSlice),
    buildFlatGrid(data.fullSlice, {
      label: "full slice",
      cellPx: sliceCellPx,
      cellGap: 0,
      showNumbers: false,
      fontName: opts.fontName,
      highlight: { r0: data.cropOffset, c0: data.cropOffset, rows: opts.size, cols: opts.size },
    }),
  ], opts.fontName, 4);

  stageFrames.rho_slice = wrapStage("ρ slice", [
    makeDensityGrid(data.slice, "ρ slice"),
  ], opts.fontName, 4);

  const halfNodes = [
    makeDensityGrid(data.h1, "h₁", "halfmap_h1"),
    makeDensityGrid(data.h2, "h₂", "halfmap_h2"),
  ];
  if (opts.include3d) {
    halfNodes.splice(1, 0, buildMeshTopology(data.h1, {
      label: "h₁ mesh", baseColor: PALETTE.blue, tileW: 14, tileH: 7,
      maxLift: opts.size * 2.2, sourceRows: opts.size, subdivisions: 4, smoothPasses: 2,
    }));
    halfNodes.push(buildMeshTopology(data.h2, {
      label: "h₂ mesh", baseColor: PALETTE.red, tileW: 14, tileH: 7,
      maxLift: opts.size * 2.2, sourceRows: opts.size, subdivisions: 4, smoothPasses: 2,
    }));
  }
  stageFrames.halfmaps = wrapStage("Half-maps", halfNodes, opts.fontName, 4);

  const rhoNodes = [makeDensityGrid(metrics.rho, "ρ̄")];
  if (opts.include3d) {
    rhoNodes.push(buildMeshTopology(metrics.rho, {
      label: "ρ̄ mesh",
      baseColor: PALETTE.purple,
      tileW: 14,
      tileH: 7,
      maxLift: opts.size * 2.2,
      sourceRows: opts.size,
      subdivisions: 4,
      smoothPasses: 2,
    }));
  }
  stageFrames.rho = wrapStage("ρ̄ average", rhoNodes, opts.fontName, 4);

  stageFrames.v = wrapStage("V (constraint)", [
    buildColoredGrid(metrics.Vnorm, {
      label: "V",
      cellPx: opts.cellPx,
      cellGap: opts.cellGap,
      fontName: opts.fontName,
      colorMode: "sequential_v",
    }),
  ], opts.fontName, 4);

  stageFrames.reliability = wrapStage("Reliability score", [
    buildColoredGrid(metrics.reliability, {
      label: "reliability",
      cellPx: opts.cellPx,
      cellGap: opts.cellGap,
      fontName: opts.fontName,
      colorMode: "reliability",
    }),
  ], opts.fontName, 4);

  stageFrames.build_zones = wrapStage("Build zones", [
    buildColoredGrid(metrics.buildZones, {
      label: "zones",
      cellPx: opts.cellPx,
      cellGap: opts.cellGap,
      fontName: opts.fontName,
      colorMode: "build_zone",
    }),
  ], opts.fontName, 4);

  const wrapper = figma.createFrame();
  wrapper.name = "cryo-em-pipeline (seed " + opts.seed + ")";
  wrapper.layoutMode = "NONE";
  wrapper.clipsContent = false;
  wrapper.fills = [{ type: "SOLID", color: { r: 0.98, g: 0.98, b: 0.99 }, opacity: 1 }];
  wrapper.strokes = [{ type: "SOLID", color: { r: 0.88, g: 0.88, b: 0.9 } }];
  wrapper.strokeWeight = 1;
  wrapper.cornerRadius = 8;

  /** @type {FrameNode[]} */
  const ordered = [];
  for (let i = 0; i < THESIS_PIPELINE.length; i++) {
    const frame = stageFrames[THESIS_PIPELINE[i]];
    if (frame) ordered.push(frame);
  }

  let x = 20;
  const y = 20;
  const arrowGap = 36;
  let maxStageH = 0;

  for (let i = 0; i < ordered.length; i++) {
    const stage = ordered[i];
    stage.x = x;
    stage.y = y;
    wrapper.appendChild(stage);
    maxStageH = Math.max(maxStageH, stage.height);

    if (i < ordered.length - 1) {
      const yMid = y + stage.height * 0.55;
      const x0 = x + stage.width + 6;
      const x1 = x + stage.width + arrowGap - 6;
      addFlowArrow(wrapper, x0, yMid, x1, yMid);
      x += stage.width + arrowGap;
    } else {
      x += stage.width + 20;
    }
  }

  wrapper.resize(x, y + maxStageH + 24);
  wrapper.x = origin.x;
  wrapper.y = origin.y;
  figma.currentPage.appendChild(wrapper);
  return wrapper;
}

/**
 * @param {string} text
 * @param {FontName} fontName
 */
function createCaption(text, fontName) {
  const label = figma.createText();
  label.fontName = fontName;
  label.fontSize = 10;
  label.characters = text.toUpperCase();
  label.fills = [{ type: "SOLID", color: { r: 0.22, g: 0.22, b: 0.24 } }];
  return label;
}

// ─── Isometric 3D ────────────────────────────────────────────────────────────

function addIsoFace(parent, pts, color, opacity) {
  let path = "";
  for (let i = 0; i < pts.length; i++) {
    path += (i === 0 ? "M" : "L") + " " + pts[i][0] + " " + pts[i][1] + " ";
  }
  path += "Z";
  const vec = figma.createVector();
  vec.vectorPaths = [{ windingRule: "NONZERO", data: path }];
  vec.fills = [solidFill(color, opacity)];
  vec.strokes = [];
  parent.appendChild(vec);
}

/**
 * @param {number} n
 */
function roundPathCoord(n) {
  return Math.round(n * 10) / 10;
}

/**
 * Catmull–Rom spline as one continuous cubic-bezier path through points.
 * @param {Array<{ x: number, y: number }>} points
 */
function catmullRomBezierPath(points) {
  if (points.length === 0) return "";
  if (points.length === 1) {
    return "M " + roundPathCoord(points[0].x) + " " + roundPathCoord(points[0].y);
  }
  let d = "M " + roundPathCoord(points[0].x) + " " + roundPathCoord(points[0].y);
  if (points.length === 2) {
    return d + " L " + roundPathCoord(points[1].x) + " " + roundPathCoord(points[1].y);
  }
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[Math.max(0, i - 1)];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[Math.min(points.length - 1, i + 2)];
    const cp1x = roundPathCoord(p1.x + (p2.x - p0.x) / 6);
    const cp1y = roundPathCoord(p1.y + (p2.y - p0.y) / 6);
    const cp2x = roundPathCoord(p2.x - (p3.x - p1.x) / 6);
    const cp2y = roundPathCoord(p2.y - (p3.y - p1.y) / 6);
    d += " C " + cp1x + " " + cp1y + " " + cp2x + " " + cp2y + " " +
      roundPathCoord(p2.x) + " " + roundPathCoord(p2.y);
  }
  return d;
}

/**
 * One vector node: each subpath is a full row or column bezier curve (uniform stroke).
 * @param {FrameNode} parent
 * @param {string[]} pathDatas
 * @param {RGB} color
 * @param {number} weight
 * @param {number} opacity
 */
function addIsoMeshGrid(parent, pathDatas, color, weight, opacity) {
  if (pathDatas.length === 0) return;
  const vec = figma.createVector();
  vec.name = "mesh grid";
  vec.vectorPaths = pathDatas.map(function (data) {
    return { windingRule: "NONZERO", data: data };
  });
  vec.strokes = [{ type: "SOLID", color: color, opacity: opacity }];
  vec.strokeWeight = weight;
  vec.strokeCap = "ROUND";
  vec.strokeJoin = "ROUND";
  vec.fills = [];
  parent.appendChild(vec);
}

/**
 * Isometric wireframe mesh on normalized 0–1 terrain (0 = base, 1 = peak).
 * @param {number[][]} values
 * @param {{ label: string, baseColor: RGB, threshold?: number, tileW: number, tileH: number, maxLift?: number, subdivisions?: number, smoothPasses?: number, sourceRows?: number, strokeWeight?: number }} opts
 */
function buildMeshTopology(values, opts) {
  const sourceRows = opts.sourceRows != null ? opts.sourceRows : values.length;
  const subdivisions = opts.subdivisions != null ? opts.subdivisions : 4;
  const grid = prepareMeshGrid(values, {
    subdivisions: subdivisions,
    smoothPasses: opts.smoothPasses,
  });
  const rows = grid.length;
  const cols = grid[0].length;
  const wx = opts.tileW / (2 * subdivisions);
  const wy = opts.tileH / (2 * subdivisions);
  const threshold = opts.threshold != null ? opts.threshold : 0.03;
  const maxLift = opts.maxLift != null ? opts.maxLift : sourceRows * 2.2;
  const strokeColor = opts.baseColor;
  const strokeWeight = opts.strokeWeight != null ? opts.strokeWeight : 0.85;
  const strokeOpacity = 0.52;

  const frame = figma.createFrame();
  frame.name = opts.label;
  frame.layoutMode = "NONE";
  frame.clipsContent = false;
  frame.fills = [];

  /**
   * @param {number} r
   * @param {number} c
   */
  function vertex(r, c) {
    const v = grid[r][c];
    return {
      x: (c - r) * wx,
      y: (c + r) * wy - v * maxLift,
    };
  }

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  function trackPoint(p) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }

  /** @type {string[]} */
  const pathDatas = [];

  for (let r = 0; r < rows; r++) {
    let hasSignal = false;
    /** @type {Array<{ x: number, y: number }>} */
    const pts = [];
    for (let c = 0; c < cols; c++) {
      if (grid[r][c] >= threshold) hasSignal = true;
      const p = vertex(r, c);
      pts.push(p);
      trackPoint(p);
    }
    if (!hasSignal) continue;
    pathDatas.push(catmullRomBezierPath(pts));
  }

  for (let c = 0; c < cols; c++) {
    let hasSignal = false;
    /** @type {Array<{ x: number, y: number }>} */
    const pts = [];
    for (let r = 0; r < rows; r++) {
      if (grid[r][c] >= threshold) hasSignal = true;
      const p = vertex(r, c);
      pts.push(p);
      trackPoint(p);
    }
    if (!hasSignal) continue;
    pathDatas.push(catmullRomBezierPath(pts));
  }

  addIsoMeshGrid(frame, pathDatas, strokeColor, strokeWeight, strokeOpacity);

  if (!Number.isFinite(minX)) {
    minX = 0;
    minY = 0;
    maxX = 80;
    maxY = 60;
  }

  const pad = 12;
  frame.resize(Math.max(maxX - minX + pad * 2, 40), Math.max(maxY - minY + pad * 2, 40));

  for (let i = 0; i < frame.children.length; i++) {
    const child = frame.children[i];
    if ("x" in child) {
      child.x -= minX - pad;
      child.y -= minY - pad;
    }
  }

  return frame;
}

// ─── Scene assembly ──────────────────────────────────────────────────────────

function findClearOrigin() {
  let maxX = 0;
  let minY = 0;
  for (let i = 0; i < figma.currentPage.children.length; i++) {
    const node = figma.currentPage.children[i];
    if ("x" in node && "width" in node) {
      maxX = Math.max(maxX, node.x + node.width);
      if (minY === 0 || node.y < minY) minY = node.y;
    }
  }
  return { x: maxX > 0 ? maxX + 64 : 80, y: minY || 80 };
}

// ─── Message handlers ────────────────────────────────────────────────────────

figma.ui.onmessage = async function (msg) {
  try {
    if (msg.type === "applyGridOperation") {
      const selection = figma.currentPage.selection.filter(function (n) {
        return n.type === "FRAME" || n.type === "GROUP" || n.type === "COMPONENT" || n.type === "INSTANCE";
      });
      if (selection.length < 1) {
        throw new Error("Select one input grid first (e.g. ρ̄ or V).");
      }

      const op = msg.operation;
      if (!GRID_OPERATION_SPECS[op]) {
        throw new Error("Unknown operation: " + op);
      }

      const fontName = await loadLabelFont();
      const built = buildDerivedGridFromSelection(selection[0], op, {
        maskThreshold: msg.maskThreshold,
        fontName: fontName,
      });

      figma.currentPage.selection = [built.outFrame];
      figma.viewport.scrollAndZoomIntoView([built.gridFrame, built.outFrame]);
      figma.ui.postMessage({
        type: "result",
        text: GRID_OPERATION_SPECS[op].label + " from \"" + built.gridFrame.name +
          "\" (" + built.rows + "×" + built.cols + ").",
      });
      return;
    }

    if (msg.type === "generateTopology") {
      const selection = figma.currentPage.selection.filter(function (n) {
        return n.type === "FRAME" || n.type === "GROUP" || n.type === "COMPONENT" || n.type === "INSTANCE";
      });
      if (selection.length < 1) {
        throw new Error("Select a grid frame first (e.g. ρ̄, h₁, or h₂).");
      }

      const source = selection[0];
      const gridFrame = resolveGridFrame(source);
      const values = gridValuesFromFrame(source);
      const rows = values.length;
      const baseColor = msg.baseColor || PALETTE.purple;
      const topo = buildMeshTopology(values, {
        label: (gridFrame.name || "grid") + " mesh",
        baseColor: baseColor,
        tileW: 10,
        tileH: 5,
        maxLift: rows * 2.2,
        sourceRows: rows,
        subdivisions: 4,
        smoothPasses: 2,
      });

      placeBeside(gridFrame, topo, 12);
      figma.currentPage.selection = [topo];
      figma.viewport.scrollAndZoomIntoView([gridFrame, topo]);
      figma.ui.postMessage({
        type: "result",
        text: "Created slope mesh from \"" + gridFrame.name + "\" (" + rows + "×" + values[0].length + ").",
      });
      return;
    }

    if (msg.type !== "generatePipeline") return;

    const fontName = await loadLabelFont();
    const size = msg.gridSize || DEFAULT_GRID_SIZE;
    const seed = msg.seed || Math.floor(Math.random() * 99999);
    const disagreement = msg.disagreement != null ? msg.disagreement : 0.55;
    const data = generateProteinSceneData(size, seed, disagreement, msg.archetype);
    const metrics = computePipelineMetrics(data);
    const wrapper = await generatePipelineDiagram(data, metrics, {
      seed: seed,
      size: size,
      include3d: msg.include3d === true,
      cellPx: DEFAULT_CELL_PX,
      cellGap: DEFAULT_CELL_GAP,
      fontName: fontName,
    });

    figma.currentPage.selection = [wrapper];
    figma.viewport.scrollAndZoomIntoView([wrapper]);
    figma.ui.postMessage({
      type: "result",
      text: "Generated pipeline → build zones (seed " + seed + ").",
    });
  } catch (err) {
    figma.ui.postMessage({
      type: "error",
      text: err instanceof Error ? err.message : String(err),
    });
  }
};
