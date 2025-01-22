% Fig. 12

clc, clear, close all;


%% load data
jsonText = jsondecode(fileread('size_and_contrast_curve.json'));
hCurve = jsonText.hCurve;
wCurve = jsonText.wCurve;
contrastCurve = jsonText.contrastCurve;

jsonText = jsondecode(fileread('velocity_curve.json'));
curveFracSTMD = jsonText.curveFracSTMD;
curveSTMDNet = jsonText.curveSTMDNet;

sizeList = [0.1:0.1:0.9, 1:10];
LuminanceList = 0:0.1:1;
vList = [1, 4, 7, 8, 9, 10:10:90, 100:25:300, 500, 1000];
tauList = [1:6, 8:2:20];

%% size curve path
h1 = figure(1);
plot(sizeList, hCurve/max(hCurve), 'b', 'DisplayName', 'height curve', 'LineWidth', 2);
hold on;
plot(sizeList, wCurve/max(wCurve), 'magenta', 'DisplayName', 'width curve', 'LineWidth', 1.5);
set(gca, 'XScale', 'log');
xlabel('Size (\circ)', 'FontSize', 12);
ylabel('Peak Model Response (normalized)', 'FontSize', 12);
xlim([0.1, 10]);
ylim([0, 1]);
legend('FontSize', 12);
set(gca, 'XTick', [0.1, 1, 10], 'XTickLabel', {'0.1', '1', '10'});
set(gca, 'YTick', [0, 0.2, 0.4, 0.6, 0.8, 1], 'YTickLabel', {'0', '0.2', '0.4', '0.6', '0.8', '1'});
savefig(h1, 'size_curve.fig');

%% luminance curve path
h2 = figure(2);
curve = contrastCurve/max(contrastCurve);
plot(LuminanceList, flip(curve), 'r', 'DisplayName', 'contrast curve', 'LineWidth', 1.5);
xlabel('Webor Contrast', 'FontSize', 12);
ylabel('Peak Model Response (normalized)', 'FontSize', 12);
xlim([0, 1]);
ylim([0, 1]);
legend('Location', 'northwest', 'FontSize', 12);
set(gca, 'XTick', [0, 0.2, 0.4, 0.6, 0.8, 1], 'XTickLabel', {'0', '0.2', '0.4', '0.6', '0.8', '1'});
set(gca, 'YTick', [0, 0.2, 0.4, 0.6, 0.8, 1], 'YTickLabel', {'0', '0.2', '0.4', '0.6', '0.8', '1'});
savefig(h2, 'contract_curve.fig');

%% velocity curve path
h3 = figure(3);
hold on;
for i = 1:size(curveFracSTMD, 1)
    if mod(i, 4) == 1
        curveF = curveFracSTMD(i, :) / max(curveFracSTMD(i, :));
        plot(vList, curveF, 'DisplayName', sprintf('EMD-tau=%d', tauList(i)), 'LineWidth', 1);
    end
end

curveB = curveSTMDNet / max(curveSTMDNet);
plot(vList, curveB, 'r-*', 'LineWidth', 2, 'MarkerEdgeColor', 'k', 'DisplayName', 'Proposed');
set(gca, 'XScale', 'log');
xlim([1, 1000]);
ylim([0, 1]);
xlabel('Velocity (\circ/s)', 'FontSize', 12);
ylabel('Peak Model Response (normalized)', 'FontSize', 12);
legend('Location', 'northwest', 'FontSize', 12);
hold off;
set(gca, 'XTick', [1, 10, 100, 1000], 'XTickLabel', {'1', '10', '100', '1000'});
set(gca, 'YTick', [0, 0.2, 0.4, 0.6, 0.8, 1], 'YTickLabel', {'0', '0.2', '0.4', '0.6', '0.8', '1'});
savefig(h3, 'velocity_curve.fig');
