function show_result_matlab()
    % Fig. 11
    data = load('effective_of_FPS.mat');


    % 创建绘图
    figure('Position', [100, 100, 1800, 900]);
    tiledlayout(4, 5, 'TileSpacing', 'compact', 'Padding', 'compact'); % 4x5 图网格

    % 定义帧率和标题
    fpsList = [1000, 500, 200, 100];
    titles = {'colorImg', 'resESTMD', 'resDSTMD', 'resFracSTMD', 'resBackbonev2'};

    % 绘制第一行（GroundTruth）
    nexttile(1); % 第一行第一列
    img = data.(sprintf('fps_%d', 1000)).colorImg;
    imshow(img);
%     title('GroundTruth');

    % 循环绘制其他图
    for i = 1:numel(fpsList)
        fps = fpsList(i);
        data1 = data.(sprintf('fps_%d', fps)); % 访问当前帧率的数据

        for j = 2:5 % 跳过第一列，从第二列开始
            nexttile((i + 1) + (j-2)*5); % 计算图网格位置
            img = data1.(titles{j});

            % 处理图像：归一化 + 阈值
            img = double(img); % 确保为双精度
            if max(img(:)) > 0
                img = img / max(img(:)); % 归一化
            end
            thres = 0.25;
            img(img < thres) = 0; % 阈值处理
            img(img >= thres) = 1;

            % 显示图像
            imshow(img, []);
%             title(sprintf('%s (FPS: %d)', titles{j}(4:end), fps));
        end

    end
    
    % 添加每列标题
    colTitles = {'GroundTruth', '1000 FPS', '500 FPS', '200 FPS', '100 FPS'};
    for j = 1:5
        nexttile(j); 
        ax = gca; % 获取当前轴
        xLimits = ax.XLim; % 获取 X 轴范围
        yLimits = ax.YLim; % 获取 Y 轴范围
        text(xLimits(1) + 0.5 * (xLimits(2) - xLimits(1)), ...
            yLimits(1) - 0.1 * (yLimits(2) - yLimits(1)), colTitles{j}, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 12); % 添加竖直标题
%         annotation('textbox', [(j-1)*0.2+0.05, 0.95, 0.1, 0.05], ...
%             'String', colTitles{j}, 'EdgeColor', 'none', ...
%             'HorizontalAlignment', 'center', 'FontSize', 12);
    end
        
    % 添加每行标题
    rowTitles = {'ESTMD', 'DSTMD', 'FracSTMD', 'Proposed'};
    for i = 1:4
        % 在每行第一个子图左侧添加文本
        nexttile((i-1)*5 + 2); 
        ax = gca; % 获取当前轴
        xLimits = ax.XLim; % 获取 X 轴范围
        yLimits = ax.YLim; % 获取 Y 轴范围
        text(xLimits(1) - 0.07 * (xLimits(2) - xLimits(1)), mean(yLimits), rowTitles{i}, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 12, 'Rotation', 90); % 添加竖直标题
        if i > 1
            axis off;
        end
    end
    
    savefig('effective_of_FPS.fig');
end