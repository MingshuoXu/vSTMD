clc, clear, close all;

%% addpath
run('../add_package_path.m');

%% Import necessary packages
import smalltargetmotiondetectors.api.*;
import smalltargetmotiondetectors.util.iostream.*;

%% Model instantiation
model = instancing_model('STMDNet');

%% input
fullpath = mfilename('fullpath');
[path1,~]=fileparts(fullpath);
hSteam = VidstreamReader(fullfile(path1, 'LImodel_input.mp4'));

%% Initialize the model
model.init_config();

%% storage data along t
x0 = 155; y0 = 295;
[intensity, retina, lamina, medullaOn, medullaOff, ...
    medullaOnV, medullaOffV, lobulaM, lobulaD] = deal(zeros(320,1));

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
    intensity(t) = grayImg(x0,y0);
    retina(t) = model.retinaOpt(x0,y0);
    lamina(t) = model.laminaOpt(x0,y0);
    medullaOn(t) = max(lamina(t),0);
    medullaOff(t) = max(-lamina(t),0);
    medullaOnV(t) = model.medullaOpt{1}(x0,y0);
    medullaOffV(t) = model.medullaOpt{2}(x0,y0);
    lobulaM(t) = medullaOnV(t)*medullaOffV(t);
end

%%
figure();
x1 = 70; x2 = 270;
x = [x1:1:x2];

subplot(4,2,1);
plot(x, retina(x1:x2));
subtitle('retina');

subplot(4,2,2);
plot(x, lamina(x1:x2));
subtitle('lamina');

subplot(4,2,4);
plot(x, medullaOn(x1:x2));
subtitle('medullaOn');

subplot(4,2,6);
plot(x, medullaOff(x1:x2));
subtitle('medullaOff');

subplot(4,2,3);
plot(x, medullaOnV(x1:x2));
subtitle('medullaOnV');

subplot(4,2,5);
plot(x, medullaOffV(x1:x2));
subtitle('medullaOffV');

subplot(4,2,7);
plot(x, lobulaM(x1:x2));
subtitle('lobulaM');



