% ═══════════════════════════════════════════════════════════════════════════
% LH_schematic_v2.m
% Laplacian-Hessian decomposition — stylized isometric schematic, 4 layers
% No text is drawn here — labels are added in Figma (see notes at bottom).
% Cell values for each checkerboard layer are printed to stdout when run.
%
% Run with: octave --no-gui LH_schematic_v2.m
% ═══════════════════════════════════════════════════════════════════════════

close all; clear; clc;

GRAY_LIGHT = [0.90 0.90 0.90];
GRAY_MID   = [0.62 0.62 0.62];
GRAY_DEEP  = [0.32 0.32 0.32];
LINE_GRAY  = [0.25 0.25 0.25];

VIEW_AZ = -45;
VIEW_EL = 28;

figure('Units', 'centimeters', 'Position', [0 0 18 22], 'Color', [1 1 1]);
axes('Position', [0.06 0.05 0.88 0.90]);
hold on;

% ── Grid geometry: an N x N grid of unit cells in [-N/2, N/2]^2 ───────────
N = 4;
half = N/2;
edges = -half:half;            % cell-corner coordinates
ctr   = edges(1:end-1) + 0.5;  % cell-center coordinates

layer_spacing = 2.6;
Z_TOP = 4.6;                       % density-surface plane (wells)
Z_T   = Z_TOP - 2*layer_spacing;   % T   layer  (half-map disagreement)
Z_V   = Z_T   - layer_spacing;     % V   layer  (gradient magnitude)
Z_RHO = Z_V   - layer_spacing;     % rho layer  (raw density, bottom)

% ── Density model: two potential wells = atom sites ──────────────────────
% Deep/narrow well  -> ordered atom   (sharp density peak, high |grad|)
% Shallow/broad well -> disordered atom (broad bump, low |grad|, noisy)
res = 0.06;
tx = -half:res:half;
ty = -half:res:half;
[X, Y] = meshgrid(tx, ty);

well_deep = [-0.9  0.6];   % ordered atom site
well_shal = [ 0.8 -0.7];   % disordered atom site

r_deep = sqrt((X - well_deep(1)).^2 + (Y - well_deep(2)).^2) + eps;
r_shal = sqrt((X - well_shal(1)).^2 + (Y - well_shal(2)).^2) + eps;

depth_deep = 1.7;  width_deep = 0.55;
depth_shal = 0.7;  width_shal = 1.1;

surface_z = -depth_deep * exp(-(r_deep.^2) / (2*width_deep^2)) ...
            -depth_shal * exp(-(r_shal.^2) / (2*width_shal^2));

% Raw density rho = positive bumps (atoms sit at density maxima)
rho_field = -surface_z;

% Gradient magnitude |grad(rho)| -- the "V" (potential / sharpness) field
[drdx, drdy] = gradient(rho_field, res);
gradmag_field = sqrt(drdx.^2 + drdy.^2);

% T (half-map disagreement / reconstruction noise) is, by the LH framework,
% complementary to V: sharp/ordered regions reproduce consistently (low T,
% high V); broad/disordered regions are noisy and disagree more (high T,
% low V). Model it as the inverse of the (smoothed, normalized) V field.
gn = gradmag_field / max(gradmag_field(:));
T_field = 1 - gn;

% ── Helper: per-cell mean of a fine field, normalized to [0, 1] ──────────
function vals = cell_field(field, X, Y, edges)
  N = numel(edges) - 1;
  vals = zeros(N, N);
  for i = 1:N
    for j = 1:N
      mask = (X >= edges(i) & X < edges(i+1) & Y >= edges(j) & Y < edges(j+1));
      vals(i,j) = mean(field(mask));
    end
  end
  vals = (vals - min(vals(:))) / (max(vals(:)) - min(vals(:)));
end

V_vals   = cell_field(gradmag_field, X, Y, edges);
T_vals   = cell_field(T_field,       X, Y, edges);
rho_vals = cell_field(rho_field,     X, Y, edges);

% ── Helper: draw one flat checkerboard-style layer at height z0 ──────────
% vals: NxN matrix in [0,1] used to pick a shade from the 3-stop grayscale
function draw_layer(edges, z0, vals, GRAY_LIGHT, GRAY_MID, GRAY_DEEP, LINE_GRAY)
  N = size(vals, 1);
  for i = 1:N
    for j = 1:N
      x0 = edges(i);   x1 = edges(i+1);
      y0 = edges(j);   y1 = edges(j+1);
      v  = vals(i,j);
      if v < 0.34
        col = GRAY_LIGHT;
      elseif v < 0.67
        col = GRAY_MID;
      else
        col = GRAY_DEEP;
      end
      fill3([x0 x1 x1 x0], [y0 y0 y1 y1], [z0 z0 z0 z0], col, ...
            'EdgeColor', LINE_GRAY, 'LineWidth', 0.6, 'FaceAlpha', 1.0);
    end
  end
end

draw_layer(edges, Z_T,   T_vals',   GRAY_LIGHT, GRAY_MID, GRAY_DEEP, LINE_GRAY);
draw_layer(edges, Z_V,   V_vals',   GRAY_LIGHT, GRAY_MID, GRAY_DEEP, LINE_GRAY);
draw_layer(edges, Z_RHO, rho_vals', GRAY_LIGHT, GRAY_MID, GRAY_DEEP, LINE_GRAY);

% ── Top layer: density surface with the two potential wells ──────────────
% Solid filled surface, no grid — smooth as computed (full resolution)
surf(X, Y, surface_z + Z_TOP, ...
     'FaceColor', GRAY_LIGHT, 'EdgeColor', 'none', 'FaceAlpha', 1.0);

% Contour lines tracing the valleys (the wells), instead of a grid
contour3(X, Y, surface_z + Z_TOP, 8, 'LineColor', LINE_GRAY, 'LineWidth', 0.6);

% Mark the two atom sites at the bottom of their wells
plot3(well_deep(1), well_deep(2), -depth_deep + Z_TOP, 'o', 'MarkerSize', 5, ...
      'MarkerFaceColor', LINE_GRAY, 'MarkerEdgeColor', [1 1 1], 'LineWidth', 0.6);
plot3(well_shal(1), well_shal(2), -depth_shal + Z_TOP, 'o', 'MarkerSize', 5, ...
      'MarkerFaceColor', LINE_GRAY, 'MarkerEdgeColor', [1 1 1], 'LineWidth', 0.6);

% ── Dashed drop-lines through the THREE flat layers only ─────────────────
% (explicitly not connecting up to the density-surface layer)
corners = [edges(1) edges(1); edges(1) edges(end); ...
           edges(end) edges(1); edges(end) edges(end)];
for k = 1:size(corners,1)
  cx = corners(k,1); cy = corners(k,2);
  plot3([cx cx], [cy cy], [Z_RHO Z_T], '--', ...
        'Color', LINE_GRAY, 'LineWidth', 0.7);
end

% ── Left-side layer labels (Helvetica) ────────────────────────────────────
lx = -half - 1.2;
ly =  half + 0.2;
LABEL_COL = [0.20 0.20 0.20];
text(lx, ly, Z_TOP, 'density  \rho', ...
     'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
     'Color', LABEL_COL, 'HorizontalAlignment', 'right');
text(lx, ly, Z_T,   'T   (half-map disagreement)', ...
     'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
     'Color', LABEL_COL, 'HorizontalAlignment', 'right');
text(lx, ly, Z_V,   'V   (gradient magnitude)', ...
     'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
     'Color', LABEL_COL, 'HorizontalAlignment', 'right');
text(lx, ly, Z_RHO, '\rho   (raw density)', ...
     'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', ...
     'Color', LABEL_COL, 'HorizontalAlignment', 'right');

view(VIEW_AZ, VIEW_EL);
axis off;
axis equal;

% ── Export ────────────────────────────────────────────────────────────────
print('-dsvg', 'LH_schematic_v2.svg');
print('-dpdf', '-bestfit', 'LH_schematic_v2.pdf');
print('-dpng', '-r300',    'LH_schematic_v2.png');

fprintf('Done. Saved LH_schematic_v2.svg / .pdf / .png\n\n');

% ── Print cell values for manual labeling (rows = i (x), cols = j (y)) ───
fprintf('--- T  (half-map disagreement) layer, normalized [0,1] ---\n');
disp(T_vals);
fprintf('--- V  (gradient magnitude) layer, normalized [0,1] ---\n');
disp(V_vals);
fprintf('--- rho (raw density) layer, normalized [0,1] ---\n');
disp(rho_vals);

% ═══════════════════════════════════════════════════════════════════════════
% LABEL PLACEMENT NOTES (add these as text objects in Figma)
%
% Layer order, top to bottom:
%   1. density surface  ρ(x,y)  — mesh with two wells (atom sites)
%   2. T   — half-map disagreement / reconstruction noise
%   3. V   — gradient magnitude / potential steepness
%   4. ρ   — raw density (per-cell average), bottom layer
%
% Left-side stack labels (align on one x so they read top-to-bottom):
%   "density ρ"                 — left of layer 1
%   "T  (half-map agreement)"   — left of layer 2
%   "V  (gradient magnitude)"   — left of layer 3
%   "ρ  (raw density)"          — left of layer 4
%
% Numeric values for each cell are printed to stdout (run the script to see
% them) — place each number in the center of its isometric cell. Matrix rows
% (i) correspond to the x-direction, columns (j) to the y-direction; cell
% (i,j) occupies x in [edges(i), edges(i+1)], y in [edges(j), edges(j+1)].
%
% Title "Laplacian–Hessian decomposition of cryo-EM density" — centered above.
% ═══════════════════════════════════════════════════════════════════════════
