clc, clear, close all;

%% addpath
run('../add_package_path.m');

%% Import necessary packages
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.util.iostream.*;

%% Model instantiation
model = instancing_model('Backbonev2');

%% input
fullpath = mfilename('fullpath');
[path1,~]=fileparts(fullpath);
hSteam = VidstreamReader(fullfile(path1, 'direction_modelling_input.mp4'));

%% Initialize the model
model.init_config();

%% Run inference
t = 0;
while hSteam.hasFrame
    t = t + 1;
    % Get the next frame from the input source
    [grayImg, colorImg] = hSteam.get_next_frame();
    
    % Perform inference using the model
    model.retinaOpt = model.hRetina.process(grayImg);
    model.laminaOpt = model.hLamina.process(model.retinaOpt);
    
    model.hMedulla.process(model.laminaOpt);
    model.medullaOpt = model.hMedulla.Opt;
    
    [model.lobulaOpt, model.modelOpt.direction] ...
        = model.hLobula.process(...
        model.medullaOpt{1}, ...
        model.medullaOpt{2}, ...
        model.laminaOpt);
    
    model.modelOpt.response = model.lobulaOpt;
    
    %%
    x0 = 125;
    y1 = 445; y2 = 462;
    if t == 50-1
        last_retina = model.retinaOpt(x0,y1:y2);
    elseif t == 50
        intensity = grayImg(x0,y1:y2);
        retina = model.retinaOpt(x0,y1:y2);
        lamina = model.laminaOpt(x0,y1:y2);
        medullaOn = max(lamina,0);
        medullaOff = max(-lamina,0);
        medullaOnV = model.medullaOpt{1}(x0,y1:y2);
        medullaOffV = model.medullaOpt{2}(x0,y1:y2);
        lobulaM = medullaOnV.*medullaOffV;
        
        on = medullaOnV; on(medullaOn>0) = 1./medullaOnV(medullaOn>0);
        off = medullaOffV; off(medullaOff>0) = 1./medullaOffV(medullaOff>0);
        lobulaD = on.*off;
        break
    end
end

%%
figure();

x = [y1:1:y2];

subplot(6,1,1);
hold on;
plot(x, retina, 'Color', [0.8500, 0.3250, 0.0980]);
plot(x, last_retina, 'k--');
subtitle('retina');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(6,1,2);
plot(x, lamina);
subtitle('lamina');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(6,1,3);
hold on;
plot(x, medullaOn, 'r*-.');
plot(x, medullaOff, 'bo-.');
subtitle('medulla');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(6,1,4);
hold on;
plot(x, medullaOnV, 'r');
plot(x, medullaOffV, 'b');
subtitle('medulla-V');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(6,1,5);
plot(x, lobulaM, 'k');
subtitle('lobulaM');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(6,1,6);
plot(x, lobulaD, 'Color', [0.4940, 0.1840, 0.5560]);
subtitle('lobulaD');
set(gca, 'XColor', 'none', 'YColor', 'none');

%% zoom-in
yy1 = 10 ; yy2 = 12;
medullaOnVSlice = medullaOnV(yy1:yy2);
medullaOffVSlice = medullaOffV(yy1:yy2);

figure()

subplot(4,1,1);
hold on;
plot(yy1:yy2, medullaOn(yy1:yy2), 'r*-.');
plot(yy1:yy2, medullaOff(yy1:yy2), 'bo-.');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(4,1,2);
hold on;
plot(yy1:yy2, medullaOnVSlice, 'r');
plot(yy1:yy2, medullaOffVSlice, 'b');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(4,1,3);
semilogy(yy1:yy2, 1./medullaOnVSlice, 'r');
hold on;
semilogy(yy1:yy2, medullaOffVSlice, 'b');
set(gca, 'XColor', 'none', 'YColor', 'none');

subplot(4,1,4);
plot(yy1:yy2, 1./medullaOnVSlice.*medullaOffVSlice, 'Color', [0.4940, 0.1840, 0.5560]);
set(gca, 'XColor', 'none', 'YColor', 'none');

