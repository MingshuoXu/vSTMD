clc, clear, close all;


ALPHA_LIST = 0.1:0.1:1.0;
G_LEAK_LIST = 0:0.1:1.0;

[X, Y] = meshgrid(G_LEAK_LIST, ALPHA_LIST);

filePath = fullfile(fileparts(mfilename('fullpath')), ...
    'parameter_analysis.json');
data = jsondecode(fileread(filePath));
aucCurve = data.aucCurve;
arCurve = data.arCurve;
apCurve = data.apCurve;
f1Curve = data.f1Curve;

aucCurve1 = zeros(length(ALPHA_LIST), length(G_LEAK_LIST));
arCurve1 = zeros(length(ALPHA_LIST), length(G_LEAK_LIST));
apCurve1 = zeros(length(ALPHA_LIST), length(G_LEAK_LIST));
f1Curve1 = zeros(length(ALPHA_LIST), length(G_LEAK_LIST));

curves = {aucCurve, arCurve, apCurve, f1Curve};
for i = 1:length(curves)
    curve = curves(i);
    for j = 1:length(ALPHA_LIST)
        for k = 1:length(G_LEAK_LIST)
            gLeakData = curve{1}(j, k, :);
            if i == 1
                aucCurve1(j, k) = mean(gLeakData);
            elseif i == 2
                arCurve1(j, k) = mean(gLeakData);
            elseif i == 3
                apCurve1(j, k) = mean(gLeakData);
            elseif i == 4
                f1Curve1(j, k) = mean(gLeakData);
            end
        end
    end
end

figure('Position', [50, 100, 1000, 300]);
tiledlayout(1, 2);

colormap(hot);
clim([0, 1]);


curvesName = {'AUC', 'AR', 'AP', 'F_{1} score'};
for i = [1, 4]
    if i == 1
        data = aucCurve1;
    elseif i == 2
        data = arCurve1;
    elseif i == 3
        data = apCurve1;
    elseif i == 4
        data = f1Curve1;
    end
    nexttile
    contourf(X, Y, data, 6);
    title(curvesName{i});
    % set(gca, 'XTick', 1:2:length(G_LEAK_LIST), ...
    %     'XTickLabel', arrayfun(@(x) sprintf('%.1f', x), G_LEAK_LIST(1:2:length(G_LEAK_LIST)),...
    %     'UniformOutput', false));
    % set(gca, 'YTick', 1:2:length(ALPHA_LIST), ...
    %     'YTickLabel', arrayfun(@(x) sprintf('%.1f', x), ALPHA_LIST(1:2:length(ALPHA_LIST)),...
    %     'UniformOutput', false));
    xlabel('gLeak');
    ylabel('alpha');
end

cb = colorbar('Ticks', [0, 0.2, 0.4, 0.6, 0.8, 1],...
    'TickLabels',{'0', '0.2', '0.4', '0.6', '0.8', '1'});
cb.Layout.Tile = 'east';

