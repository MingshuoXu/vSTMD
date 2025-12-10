clc; clear; close all;

%%

% dataset path
ristDatasetPath = fullfile('D:', 'STMD_Dataset', 'RIST');

% dataset information


datasetInfo = {
    'GX010071-1', 1:1300;
    'GX010220-1', 1:1300;
    'GX010228-1', 1:1300;
    'GX010230-1', 1:2400;
    'GX010231-1', 1:2400;
    'GX010241-1', 1:3600;
    'GX010250-1', 1:2000;
    'GX010266-1', 1:2400;
    'GX010290-1', 1:1300;
    'GX010291-1', 1:1300;
    'GX010303-1', 1:2400;
    'GX010307-1', 1:1000;
    'GX010315-1', 1:1000;
    'GX010321-1', 1:1000;
    'GX010322-1', 1:1300;
    'GX010327-1', 1:900;
    'GX010335-1', 1:1300;
    'GX010336-1', 1:1000;
    'GX010337-1', 1:700;
};

currPth = fileparts(mfilename('fullpath'));
file_path = fullfile(currPth, 'visulize_data.json');


opticflowModelList = {'RAFT', 'MemFlow', 'StreamFlow', 'DpFlow', 'FlowDiffuser'};

directionalStmdList = {'STMDPlus', 'ApgSTMD', 'vSTMD', 'vSTMD_F'};

modelList = [opticflowModelList, directionalStmdList];


fileData = jsondecode(fileread(file_path));

%%
figure('Position', [100, 100, 1500, 700]);
tiledlayout(7, 6, 'TileSpacing', 'compact', 'Padding', 'compact'); 

%% main

i = 1;
for i = 1:19
    datasetName = datasetInfo{i, 1};
    frame0 = datasetInfo{i, 2}(1); frameEnd = datasetInfo{i, 2}(end);

    % Display the raw image
    rawImg = read_last_img(fullfile(ristDatasetPath, datasetName, sprintf('%s.mp4', datasetName)),...
        frameEnd);
    nexttile;
    imshow(rawImg);
    ylabel(gca, {sprintf('%s', datasetName)}, ...
        'Rotation', 90, ...  % 水平显示
        'HorizontalAlignment', 'right', ...
        'VerticalAlignment', 'middle', ...
        'Position', [-0.3 0.5 0], ...
        'Margin', 10); % 调整位置

    [sizeM, sizeN, ~] = size(rawImg);

    % groundtruth
    GT = fileData.(sprintf('%s_1_groundtruth', datasetName(1:end-2)));
    nexttile;
    axis([1 sizeN 1 sizeM]); 

    custom_plot(GT.location(frame0:frameEnd,1),...
        GT.location(frame0:frameEnd,2), ...
        GT.direction(frame0:frameEnd));
    
    if i == 1; title('ground turth'); end
   
end


function img = read_last_img(pth, frameNum)
    v = VideoReader(pth);
    img = read(v, frameNum);

end

function custom_plot(x, y, z)

    hold on;
    cmap = hsv(256); % 使用HSV色图，非常适合相位数据

    % 归一化Z到1-256
    z_norm = round((z - 0)/(2*pi) * 255 + 1);
    
    scatter(x, y, 20, z, 'filled');
    % % 绘制彩色线段
    % for ii = 1:length(x)
    %     plot(x(ii), y(ii), ...
    %          'Color', cmap(z_norm(ii),:), 'LineWidth', 2);
    % end
    set(gca, 'XTick', [], 'YTick', []); 

end