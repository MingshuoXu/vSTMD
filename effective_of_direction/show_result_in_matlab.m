function show_result_in_matlab()
    % Fig. 8

    close all;
    
    timeEnd = 995;

    data = load('error_direction.mat');
    bgImg = imread('PanoramaStimuli1000.tif');
    
    errSTMDNet = data.errSTMDNet;
    errD = data.errD;
    posiGT = data.posiGT;

    errSTMDNet_plot = errSTMDNet(1:timeEnd);
    errSTMDNet_plot(isnan(errSTMDNet_plot)) = -pi;  

    errD_plot = errD(1:timeEnd);
    errD_plot(isnan(errD_plot)) = -pi;  

    h = figure('Position', [100, 100, 500, 650]);  
    tiledlayout(2,1, 'Padding', 'compact','TileSpacing', 'compact');
    
    cmap = colormap('jet'); 
    bounds = [-pi, -pi/2, -pi/4, -pi/8, 0, pi/8, pi/4, pi/2, pi];  
    clim([-pi, pi]);

    colormap(cmap);
    
    % subfig 1
    nexttile
    imshow(bgImg);
    hold on;
    scatter(posiGT(1:timeEnd, 2), posiGT(1:timeEnd, 1), 20, errSTMDNet_plot, 'filled');
    title('Direction Error (Proposed)');
    xlim([0 470]);
    ylim([0 310]);

    
    % Dubfig 2
    nexttile;
    imshow(bgImg);
    hold on;
    scatter(posiGT(1:timeEnd, 2), posiGT(1:timeEnd, 1), 20, errD_plot, 'filled');
    title('Direction Error (DSTMD)');
    xlim([0 470]);
    ylim([0 310]);
   
    cb = colorbar('Ticks', bounds, ...
         'TickLabels', {'$-\pi$', '$-\pi/2$', '$-\pi/4$', '$-\pi/8$', '$0$', '$\pi/8$', '$\pi/4$', '$\pi/2$', '$\pi$'});
    set(cb, 'TickLabelInterpreter', 'latex');

    cb.Layout.Tile = 'east';

    savefig('error_direction.fig');
end
