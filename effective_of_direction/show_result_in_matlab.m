function show_reslut_in_matlab()
    % Fig. 10

    close all;
    
    timeEnd = 995;

    data = load('error_direction.mat');
    bgImg = imread('PanoramaStimuli1000.tif');
    
    errB = data.errB;
    errD = data.errD;
    posiGT = data.posiGT;

    % 对NaN值进行处理
    errB_plot = errB(1:timeEnd);
    errB_plot(isnan(errB_plot)) = -pi;  % 将NaN值替换为一个较大的值

    errD_plot = errD(1:timeEnd);
    errD_plot(isnan(errD_plot)) = -pi;  % 将NaN值替换为一个较大的值

    % 创建子图
    figure('Position', [100, 100, 1400, 400]);  % 设置画布大小
    tiledlayout(1,2);
    
    cmap = colormap('jet'); 
%     cmap(121:135,:) = max(cmap(121:135,:)*3, 1);
%     cmap = flipud(cmap);  % 反转颜色映射
    bounds = [-pi, -pi/2, -pi/4, -pi/8, pi/8, pi/4, pi/2, pi];  % 误差的范围
    caxis([-pi, pi]);

    % 创建新的颜色映射，黑色边界
    colormap(cmap);
    
    % 子图1：绘制 Backbonev2 的方向误差
    nexttile
    imshow(bgImg);
    hold on;
    scatter(posiGT(1:timeEnd, 2), posiGT(1:timeEnd, 1), 20, errB_plot, 'filled');
%     xlabel('X axis');
%     ylabel('Y axis');
    title('Direction Error (Proposed)');
    xlim([0 470]);
    ylim([0 310]);

    
    % 子图2：绘制 DSTMD 的方向误差
    nexttile;
    imshow(bgImg);
    hold on;
    scatter(posiGT(1:timeEnd, 2), posiGT(1:timeEnd, 1), 20, errD_plot, 'filled');
%     xlabel('X axis');
%     ylabel('Y axis');
    title('Direction Error (DSTMD)');
    xlim([0 470]);
    ylim([0 310]);
   
    cb = colorbar('Ticks',bounds,...
         'TickLabels',{'-pi', '-pi/2', '-pi/4', '-pi/8', 'pi/8', 'pi/4', 'pi/2', 'pi'});
    cb.Layout.Tile = 'east';
    
    savefig('error_direction.fig');
end
