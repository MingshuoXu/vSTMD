% 调用主函数
clc,clear,close all;

% 读取第一个 JSON 文件
data1 = jsondecode(fileread('velocity_AUC_curve.json'));
AUCcurveESTMD = data1.AUCcurveESTMD;
AUCcurveDSTMD = data1.AUCcurveDSTMD;
AUCcurveFracSTMD = data1.AUCcurveFracSTMD;
AUCcurveSTMDNet = data1.AUCcurveSTMDNet;

% 速度范围
vList = 50:50:2000;
TAU_LIST = [1,5,11,19,29];

% 创建子图
figure;
ax1 = subplot(2, 2, 1);
ax2 = subplot(2, 2, 2);
ax3 = subplot(2, 2, 3);
ax4 = subplot(2, 2, 4);

% 绘制曲线
for i = 1:5
    plot(ax1, vList, AUCcurveESTMD(i, :), 'DisplayName', sprintf('tau=%d, mAUC = %0.2f', TAU_LIST(i), mean(AUCcurveESTMD(i, :))));
    hold(ax1, 'on');
    plot(ax2, vList, AUCcurveDSTMD(i, :), 'DisplayName', sprintf('tau=%d, mAUC = %0.2f', TAU_LIST(i), mean(AUCcurveDSTMD(i, :))));
    hold(ax2, 'on');
    plot(ax3, vList, AUCcurveFracSTMD(i, :), 'DisplayName', sprintf('tau=%d, mAUC = %0.2f', TAU_LIST(i), mean(AUCcurveFracSTMD(i, :))));
    hold(ax3, 'on');
end
plot(ax4, vList, AUCcurveSTMDNet, 'r-*', 'LineWidth', 2, 'MarkerEdgeColor', 'r',...
    'DisplayName', sprintf('without tau, mAUC = %0.2f', mean(AUCcurveSTMDNet)));

% 设置图例和轴标签
set([ax1, ax2, ax3, ax4], 'XLim', [0 2000], 'YLim', [0 1]);
xlabel(ax1, 'velocity (pixels/s)');
ylabel(ax1, 'AUC');
title(ax1, 'ESTMD');
legend(ax1);

xlabel(ax2, 'velocity (pixels/s)');
ylabel(ax2, 'AUC');
title(ax2, 'DSTMD');
legend(ax2);

xlabel(ax3, 'velocity (pixels/s)');
ylabel(ax3, 'AUC');
title(ax3, 'FracSTMD');
legend(ax3, 'Location', 'SouthEast');

xlabel(ax4, 'velocity (pixels/s)');
ylabel(ax4, 'AUC');
title(ax4, 'Proposed');
legend(ax4, 'Location', 'SouthEast');

% 显示图像
hold off;



