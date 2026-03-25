% spectrogramDatasetGeneration.m creates the spectrogram datastores for the CNN-LSTM model.
% Files are generated in "DatastoresLSTM/" folder.
%
% This script supports TWO dataset formats:
%   - Format JSON (trainingJSON1): EMG-EPN-612 original, one JSON per user.
%   - Format MAT  (trainingJSON):  New dataset, one .mat file per gesture per user.
%
% Change DATASET_FORMAT (and trainingDir accordingly) to switch between datasets.

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

laboratorio.ia@epn.edu.ec

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

Matlab 9.11.0.2022996 (R2021b) Update 4.
%}

%% DEFINE THE DIRECTORIES WHERE THE DATA WILL BE FOUND
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);

dataDir = fullfile(rootDir, 'EMG-EPN-612 dataset');

% #################################################################
% ######### SELECT DATASET FORMAT - CHANGE THIS BLOCK #############
% #################################################################
% Options:
%   'mat'  -> New dataset  (trainingJSON/,  no numbers, .mat files per gesture)
%   'json' -> Original dataset (trainingJSON1/, with number, one JSON per user)
DATASET_FORMAT = 'mat';

if isequal(DATASET_FORMAT, 'mat')
    trainingDir  = 'trainingJSON';
    datastoreOut = fullfile(rootDir, 'DatastoresLSTM');     % output folder
elseif isequal(DATASET_FORMAT, 'json')
    trainingDir  = 'trainingJSON1';
    datastoreOut = fullfile(rootDir, 'DatastoresLSTM1');    % output folder
else
    error('Unknown DATASET_FORMAT: %s. Use ''mat'' or ''json''.', DATASET_FORMAT);
end
% #################################################################
% #################################################################


%% GET THE USERS DIRECTORIES
[users, trainingPath] = Shared.getUsers(dataDir, trainingDir);
if Shared.includeTesting
    limit = length(users) - Shared.numTestUsers;
    usersTrainVal = users(1:limit, 1);
    usersTest = users(limit+1:length(users), 1);
else
    usersTrainVal = users;
end
clear dataDir trainingDir users numTestUsers limit

%% ===== JUST FOR DEBUGGING =====
% usersTrainVal = usersTrainVal(1:3);
% usersTest = usersTest(1:1);
%  ===== JUST FOR DEBUGGING =====

%% THE STRUCTURE OF THE DATASTORE IS DEFINED
categories = {'fist'; 'open'; 'pinch'; 'waveIn'; 'waveOut'};
trainingDatastore   = createDatastore(fullfile(datastoreOut, 'training'),   categories);
validationDatastore = createDatastore(fullfile(datastoreOut, 'validation'), categories);
if Shared.includeTesting
    testingDatastore = createDatastore(fullfile(datastoreOut, 'testing'), categories);
end
clear categories datastoreOut

%% GENERATION OF SPECTROGRAMS TO CREATE THE MODEL
if Shared.includeTesting
    usersSets = {usersTrainVal, 'usersTrainVal'; usersTest, 'usersTest'};
else
    usersSets = {usersTrainVal, 'usersTrainVal'};
end

for i = 1:size(usersSets, 1)
    users   = usersSets{i,1};
    usersSet = usersSets{i,2};
    fprintf('User set %s\n', usersSet)

    if isequal(usersSet, 'usersTrainVal')
        [datastore1, datastore2] = deal(trainingDatastore, validationDatastore);
    elseif isequal(usersSet, 'usersTest')
        [datastore1, datastore2] = deal(testingDatastore, testingDatastore);
    end

    parfor j = 1:length(users)
        s = sprintf('Usuario: %d / %d\n', j, length(users));
        fprintf('%s', s)

        % Read samples using the unified logic in Shared.m
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(j), DATASET_FORMAT);

        transformedSamplesTraining   = generateData(trainingSamples);
        transformedSamplesValidation = generateData(validationSamples);

        saveSampleInDatastore(transformedSamplesTraining,   users(j), 'train',      datastore1);
        saveSampleInDatastore(transformedSamplesValidation, users(j), 'validation', datastore2);

        fprintf(repmat('\b', 1, numel(s)));
    end
end
clear i j validationSamples transformedSamplesValidation users usersSet usersTrainVal usersTest datastore1 datastore2

%% INCLUDE NOGESTURE
if Shared.includeTesting
    datastores = {trainingDatastore; validationDatastore; testingDatastore};
else
    datastores = {trainingDatastore; validationDatastore};
end
noGestureFramesPerSample = cell(length(datastores), 1);
clear trainingSamples transformedSamplesTraining trainingDatastore validationDatastore testingDatastore

%% CALCULATE THE NUMBER OF FRAMES IN A SEQUENCE FOR EACH DATASTORE
parfor i = 1:length(datastores)
    fds = fileDatastore(datastores{i,1}, ...
        'ReadFcn',@Shared.readFile, 'IncludeSubfolders',true);

    if isequal(Shared.NOGESTURE_FILL, 'all')
        numFiles = length(fds.Files);
        numFramesSamples = zeros(numFiles, 1);
        for j = 1:numFiles
            frames = load(fds.Files{j,1}).data.sequenceData;
            numFramesSamples(j,1) = length(frames);
        end
        noGestureFramesPerSample{i,1} = round(mean(numFramesSamples));

    elseif isequal(Shared.NOGESTURE_FILL, 'some')
        labels = Shared.createLabels(fds.Files, false);
        gestures = Shared.setNoGestureUse(false);
        avgNumFramesClass = zeros(length(gestures), 1);
        for j = 1:length(gestures)
            class = gestures(1, j);
            idxs = cellfun(@(label) isequal(label,class), cellstr(labels));
            filesClass = fds.Files(idxs, 1);
            numFilesClass = length(filesClass);
            numFramesSamples = zeros(numFilesClass, 1);
            for k = 1:numFilesClass
                frames = load(filesClass{k,1}).data.sequenceData;
                numFramesSamples(k,1) = length(frames);
            end
            avgNumFramesClass(j,1) = round(mean(numFramesSamples));
        end
        noGestureFramesPerSample{i,1} = [min(avgNumFramesClass), max(avgNumFramesClass)];
    end
end
clear i j k class gestures filesClass frames idxs avgNumFramesClass labels numFilesClass numFramesSamples fds

%% THE STRUCTURE OF THE DATASTORE IS DEFINED (noGesture folder)
categories = {'noGesture'};
trainingDatastore   = createDatastore(datastores{1,1}, categories);
validationDatastore = createDatastore(datastores{2,1}, categories);
if Shared.includeTesting
    testingDatastore = createDatastore(datastores{3,1}, categories);
end
clear categories datastores

%% GENERATION OF NOGESTURE SPECTROGRAMS
noGestureTraining   = noGestureFramesPerSample{1,1};
noGestureValidation = noGestureFramesPerSample{2,1};
if Shared.includeTesting
    noGestureTesting = noGestureFramesPerSample{3,1};
end

for i = 1:size(usersSets, 1)
    users   = usersSets{i,1};
    usersSet = usersSets{i,2};

    if isequal(usersSet, 'usersTrainVal')
        [sz1, sz2, ds1, ds2] = deal(noGestureTraining, noGestureValidation, ...
            trainingDatastore, validationDatastore);
    elseif isequal(usersSet, 'usersTest')
        [sz1, sz2, ds1, ds2] = deal(noGestureTesting, noGestureTesting, ...
            testingDatastore, testingDatastore);
    end

    parfor j = 1:length(users)
        % Read samples using the unified logic in Shared.m
        [trainingSamples, validationSamples] = Shared.getTrainingTestingSamples(trainingPath, users(j), DATASET_FORMAT);
        
        tTrain = generateDataNoGesture(trainingSamples, sz1);
        tVal   = generateDataNoGesture(validationSamples, sz2);
        saveSampleInDatastore(tTrain, users(j), 'validation', ds1);
        saveSampleInDatastore(tVal,   users(j), 'train',      ds2);
    end
end
clear i j ds1 ds2 sz1 sz2 noGestureTesting noGestureTraining noGestureValidation
clear testingDatastore trainingDatastore trainingPath users usersSet validationDatastore
clear transformedSamplesValidation validationSamples noGestureFramesPerSample usersSets


%% =========================================================================
%%  SPECTROGRAM GENERATION FUNCTIONS (unchanged from original)
%% =========================================================================

function datastore = createDatastore(datastore, labels)
    if ~exist(datastore, 'dir'), mkdir(datastore); end
    for i = 1:length(labels)
        path = fullfile(datastore, char(labels(i)));
        if ~exist(path, 'dir'), mkdir(path); end
    end
end

function [data, groundTruth] = generateFrames(signal, groundTruth, numGesturePoints, gestureName)
    if isequal(Shared.FILLING_TYPE_EVAL, 'before')
        noGestureInSignal = signal(~groundTruth, :);
        filling = noGestureInSignal(1:floor(Shared.FRAME_WINDOW / 2), :);
        signal = [signal; filling];
        groundTruth = [groundTruth; zeros(floor(Shared.FRAME_WINDOW / 2), 1)];
    end
    numWindows = floor((length(signal)-Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_LSTM) + 1;
    data = cell(numWindows, 3);
    data(:,2) = {'noGesture'};
    isIncluded = false(numWindows, 1);
    for i = 1:numWindows
        traslation = (i-1) * Shared.WINDOW_STEP_LSTM;
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        frameGroundTruth = groundTruth(inicio:finish);
        totalOnes = sum(frameGroundTruth == 1);
        frameSignal = signal(inicio:finish, :);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        data{i,1} = spectrograms;
        data{i,3} = timestamp;
        if totalOnes >= Shared.FRAME_WINDOW * Shared.TOLERANCE_WINDOW || ...
                totalOnes >= numGesturePoints * Shared.TOLERNCE_GESTURE_LSTM
            isIncluded(i,1) = true;
            data{i,2} = gestureName;
        end
    end
    if isequal(Shared.NOGESTURE_FILL, 'all')
        isIncluded(:,1) = true;
    elseif isequal(Shared.NOGESTURE_FILL, 'some')
        first = find(isIncluded, 1, 'first');
        last  = find(isIncluded, 1, 'last');
        for i = 1:Shared.NOGESTURE_IN_SEQUENCE
            if first - i >= 1,       isIncluded(first-i, 1) = true; end
            if last  + i <= numWindows, isIncluded(last+i,  1) = true; end
        end
    end
    data = data(isIncluded,:);
end

function transformedSamples = generateData(samples)
    noGesturePerUser = Shared.numGestureRepetitions;
    samplesKeys = fieldnames(samples);
    transformedSamples = cell(length(samplesKeys) - noGesturePerUser, 3);
    for i = noGesturePerUser + 1:length(samplesKeys)
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        groundTruth = sample.groundTruth;
        numGesturePoints = sample.groundTruthIndex(2) - sample.groundTruthIndex(1);
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        [data, newGroundTruth] = generateFrames(signal, groundTruth, numGesturePoints, gestureName);
        % Skip if no frames passed the gesture threshold
        if isempty(data)
            transformedSamples{i - noGesturePerUser, 1} = {};
            transformedSamples{i - noGesturePerUser, 2} = gestureName;
            transformedSamples{i - noGesturePerUser, 3} = [];
            continue;
        end
        transformedSamples{i - noGesturePerUser, 1} = data;
        transformedSamples{i - noGesturePerUser, 2} = gestureName;
        transformedSamples{i - noGesturePerUser, 3} = transpose(newGroundTruth);
    end
end

function saveSampleInDatastore(samples, user, type, dataStore)
    for i = 1:length(samples)
        sequenceData = samples{i,1};
        class = samples{i,2};
        % Skip samples with no frames (no window passed the gesture threshold)
        if isempty(sequenceData)
            continue;
        end
        timestamps = sequenceData(:,3);
        fileName = strcat(strtrim(user), '-', type, '-', int2str(i), '-', ...
            '[', int2str(timestamps{1,1}), '-', int2str(timestamps{end,1}), ']');
        data.sequenceData = sequenceData;
        % groundTruth column only present when samples has 3 columns (gesture samples)
        if ~isequal(class,'noGesture') && size(samples,2) >= 3
            data.groundTruth = samples{i,3};
        end
        savePath = fullfile(dataStore, char(class), fileName);
        save(savePath, 'data');
    end
end

function data = generateFramesNoGesture(signal, requestedWindows)
    numWindows = floor((length(signal) - Shared.FRAME_WINDOW) / Shared.WINDOW_STEP_LSTM) + 1;
    numWindowsFill = requestedWindows - numWindows;
    if numWindowsFill > 0
        filling = signal(1:numWindowsFill*Shared.WINDOW_STEP_LSTM, :);
        signal = [signal; filling];
    end
    data = cell(requestedWindows, 3);
    data(:,2) = {'noGesture'};
    for i = 1:requestedWindows
        traslation = (i-1) * Shared.WINDOW_STEP_LSTM;
        inicio = 1 + traslation;
        finish = Shared.FRAME_WINDOW + traslation;
        timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);
        frameSignal = signal(inicio:finish, :);
        spectrograms = Shared.generateSpectrograms(frameSignal);
        data{i,1} = spectrograms;
        data{i,3} = timestamp;
    end
end

function transformedSamples = generateDataNoGesture(samples, numFrames)
    noGesturePerUser = Shared.numGestureRepetitions;
    samplesKeys = fieldnames(samples);
    transformedSamples = cell(noGesturePerUser, 2);
    for i = 1:noGesturePerUser
        sample = samples.(samplesKeys{i});
        emg = sample.emg;
        gestureName = sample.gestureName;
        signal = Shared.getSignal(emg);
        signal = Shared.preprocessSignal(signal);
        if isequal(Shared.NOGESTURE_FILL, 'all')
            framesPerSample = numFrames;
        elseif isequal(Shared.NOGESTURE_FILL, 'some')
            rng(i);
            framesPerSample = randi(numFrames);
        end
        data = generateFramesNoGesture(signal, framesPerSample);
        transformedSamples{i,1} = data;
        transformedSamples{i,2} = gestureName;
    end
end
