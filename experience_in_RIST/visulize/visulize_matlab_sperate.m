clc; clear; close all;

%%

% dataset path
ristDatasetPath = fullfile('D:', 'STMD_Dataset', 'RIST');

% dataset information
datasetInfo = {
    'GX010230-1', 1:2400;
    'GX010241-1', 1:3600;
    'GX010266-1', 1:2400; %
    'GX010290-1', 1:1300; %
    'GX010303-1', 1:2400; %
    'GX010307-1', 1:1000; %
    'GX010337-1', 1:700; %
};

% datasetInfo = {
%     'GX010071-1', 1:1300;
%     'GX010220-1', 1:1300;
%     'GX010228-1', 1:1300;
%     'GX010230-1', 1:2400;
%     'GX010231-1', 1:2400;
%     'GX010241-1', 1:3600;
%     'GX010250-1', 1:2000;
%     'GX010266-1', 1:2400;
%     'GX010290-1', 1:1300;
%     'GX010291-1', 1:1300;
%     'GX010303-1', 1:2400;
%     'GX010307-1', 1:1000;
%     'GX010315-1', 1:1000;
%     'GX010321-1', 1:1000;
%     'GX010322-1', 1:1300;
%     'GX010327-1', 1:900;
%     'GX010335-1', 1:1300;
%     'GX010336-1', 1:1000;
%     'GX010337-1', 1:700;
% };


currPth = fileparts(mfilename('fullpath'));
file_path = fullfile(currPth, 'visulize_data.json');


opticflowModelList = {'RAFT', 'MemFlow', 'StreamFlow', 'DpFlow', 'FlowDiffuser'};

directionalStmdList = {'STMDPlus', 'ApgSTMD', 'vSTMD', 'vSTMD_F'};

modelList = [opticflowModelList, directionalStmdList];


fileData = jsondecode(fileread(file_path));

%%
% figure('Position', [100, 100, 1500, 700]);
% tiledlayout(size(datasetInfo, 1), size(modelList, 2)+3, 'TileSpacing', 'compact', 'Padding', 'compact'); 

custom_position = [200, 200, 160, 105];

%% main
axCells = {};
for i = 1:length(datasetInfo)
    datasetName = datasetInfo{i, 1};
    frame0 = datasetInfo{i, 2}(1); frameEnd = datasetInfo{i, 2}(end);

    % Display the raw image
    figure('name', sprintf('Raw Image: %s', datasetName), ...
        'Position', custom_position);
    axis tight;  % 使坐标轴紧贴数据范围

    axCells{end+1} = gca;
    rawImg = read_last_img(fullfile(ristDatasetPath, datasetName, sprintf('%s.mp4', datasetName)),...
        frameEnd);
    axCells{end+1} = nexttile;
    imshow(rawImg);
    axis image;
    % if i == 1; title('Raw Image'); end
   

    % groundtruth
    GT = fileData.(sprintf('%s_1_groundtruth', datasetName(1:end-2)));
    % axCells{end+1} = nexttile;
    figure('name', sprintf('groundturth: %s', datasetName), ...
        'Position', custom_position);
    axis tight;
    axCells{end+1} = gca;

    custom_plot(GT.location(frame0:frameEnd,1),...
        GT.location(frame0:frameEnd,2), ...
        GT.direction(frame0:frameEnd));
    
    % if i == 1; title('ground turth'); end
   
    % Loop through each model
    for j = 1:size(modelList, 2)
        modelName = modelList{j};

        resOpt = fileData.(sprintf('%s_1_%s', datasetName(1:end-2), modelName));

        % axCells{end+1} = nexttile;
        figure('name', sprintf('output: %s - %s', datasetName, modelName), ...
            'Position', custom_position);
        axis tight;
        axCells{end+1} = gca;

        largePlot = size(resOpt.directions, 1);
        if largePlot < size(resOpt.response, 1)
            frame00 = size(resOpt.response, 1) - largePlot + 1;
        else
            frame00 = 1;
        end
        custom_scatter(resOpt.response(frame00:end,1),...
            resOpt.response(frame00:end,2), ...
            resOpt.directions);
        
        % if i == 1; title(strrep(modelName, '_', '-')); end
        show_criteria(datasetName, modelName);

    end

    drawnow;
end

linkaxes([axCells{:}], 'xy');
% 获取所有 Figure 对象
allFigs = findall(0, 'Type', 'figure');

% 遍历所有 Figure
for i = 1:length(allFigs)
    fig = allFigs(i);
    
    % 获取当前 Figure 的所有坐标轴
    allAxes = findall(fig, 'Type', 'axes');
    
    % 遍历当前 Figure 的所有坐标轴
    for j = 1:length(allAxes)
        ax = allAxes(j);
        set(ax, 'LooseInset', get(ax, 'TightInset'));
    end
    
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

    % % 绘制彩色线段
    for ii = 1:length(x)-1
        plot([x(ii), x(ii+1)], [y(ii), y(ii+1)], ...
             'Color', cmap(z_norm(ii),:), 'LineWidth', 3);
    end 
    set(gca, 'XColor', 'none', 'YColor', 'none'); % 隐藏坐标轴线
    axis image;
end

function custom_scatter(x, y, z)

    hold on;
    cmap = hsv(256); % 使用HSV色图，非常适合相位数据

    % 归一化Z到1-256
    z_norm = round((z - 0)/(2*pi) * 255 + 1);

    mark = isnan(z_norm);
    x(mark)=[]; y(mark)=[]; z_norm(mark)=[];
    scatter(x, y, 5, cmap(z_norm, :), 'filled'); % 'filled' 表示实心点

    % % 绘制彩色线段
    % for ii = 1:length(x)
    %     plot(x(ii), y(ii), ...
    %          'Color', cmap(z_norm(ii),:), 'LineWidth', 2);
    % end 
    set(gca, 'XColor', 'none', 'YColor', 'none'); % 隐藏坐标轴线
    axis image;
end

function show_criteria(datasetName, modelName)
    currPth = fileparts(mfilename('fullpath'));
    file_path = fullfile(fileparts(fileparts(currPth)), ...
        'result', 'RIST_240Hz', datasetName, ...
        sprintf('%sevaluate.json', modelName) );
    data_ = jsondecode(fileread(file_path));
    
    AAE = data_.AAE;
    if isfield(data_, 'AUC')
        AUC = sprintf('%.1f%%', data_.AUC*100);
    else
        AUC = ' - ';
    end

    text(0.4, 0.85, {sprintf('AAE:%.2f; AUC:%s', AAE, AUC)}, 'sc', ...
        'HorizontalAlignment', 'center', 'FontSize', 6);
end