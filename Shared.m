%Shared is a class used to organize the shared parameters (its properties)
%and the functions (its methods) for the HGR model.
% Recommended to change only the doPostprocessing flag.

%{
Laboratorio de Inteligencia y Visión Artificial
ESCUELA POLITÉCNICA NACIONAL
Quito - Ecuador

laboratorio.ia@epn.edu.ec

"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

Matlab 9.11.0.2022996 (R2021b) Update 4.
%}

classdef Shared

    properties (Constant)
        % #################################################################
        % ######### Recommended configurations to change ##################
        % #################################################################

        % ############# Post-Processing ###################################
        % ``postprocessing``: Change this flag to either true or false
        %doPostprocessing = false;
        doPostprocessing = true;
        
        
        % ####################### Stride ###################################
        % recommended to change all vars at once
        WINDOW_STEP = 30; % To obtain the frames
        WINDOW_STEP_RECOG = 30; % 15 30
        WINDOW_STEP_LSTM = 30; % 15 30
        % #################################################################
        % #################################################################


        % Spectrogram
        FRECUENCIES = (0:12);
        WINDOW = 24;
        OVERLAPPING = floor(Shared.WINDOW * 0.5);

        % Frame
        FRAME_WINDOW = 300;
        % if frame > TOLERANCE_WINDOW || frame > TOLERNCE_GESTURE -> gesture
        TOLERANCE_WINDOW = 0.75;
        TOLERNCE_GESTURE = 0.5; % 0.75 0.25;
        
        % Recognition
        FRAME_CLASS_THRESHOLD = 0.5; % 0.75 0.25;
        % if labels > MIN_LABELS_SEQUENCE -> true
        MIN_LABELS_SEQUENCE = 4;
        FILLING_TYPE = 'before'; % 'before' 'none'
        POSTPROCESS = '1-1'; % '1-1' '1-2' '2-1'

        % Evaluation
        FILLING_TYPE_EVAL = 'none'; % 'before' 'none'

        % For LSTM
        FILLING_TYPE_LSTM = 'before'; % 'before' 'none'
        NOGESTURE_FILL = 'some' % 'some' 'all'
        NOGESTURE_IN_SEQUENCE = 3; % if 'some'
        PAD_KIND = 'shortest'; % 'shortest' 'longest'
        TOLERNCE_GESTURE_LSTM = 0.5; % 0.75 0.25;
        NUM_HIDDEN_UNITS = 128; % 128 %58(60) %27(30)

        % Samples and signals
        numSamplesUser = 150;
        numGestureRepetitions = 25;
        numChannels = 8;

        % User distribution
        includeTesting = false; %false true
        numTestUsers = 16;
    end

    methods(Static)

        % GET THE USER LIST
        function [users, dataPath] = getUsers(dataDir, subDir)
            dataPath = fullfile(dataDir, subDir);
            usersList = dir(dataPath);
            
            % Filtrar solamentente carpetas de usuarios válidas
            users = string.empty;
            for i = 1:length(usersList)
                if usersList(i).isdir && startsWith(usersList(i).name, 'user')
                    users(end+1, 1) = string(usersList(i).name);
                end
            end
            
            rng(9); % seed
            users = users(randperm(length(users)));
        end

        % GET TRAINING AND TESTING SAMPLES FOR AN USER (Unified for JSON and MAT)
        function [trainingSamples, testingSamples] = getTrainingTestingSamples(path, user, format)
            if nargin < 3, format = 'json'; end % Default to json for backward compatibility
            
            if isequal(format, 'json')
                filePath = fullfile(path, user, strcat(user, '.json'));
                jsonFile = fileread(filePath);
                jsonData = jsondecode(jsonFile);
                trainingSamples = jsonData.trainingSamples;
                testingSamples = jsonData.testingSamples;
            elseif isequal(format, 'mat')
                [trainingSamples, testingSamples] = Shared.readSamplesMAT(path, user);
            else
                error('Unknown format: %s', format);
            end
        end

        % READ SAMPLES — NEW DATASET (.mat, one file per gesture)
        function [trainingSamples, validationSamples] = readSamplesMAT(path, user)
            gestureNames = {'fist', 'open', 'pinch', 'waveIn', 'waveOut', 'relax'};
            allSamples = struct();
            idx = 1;

            for g = 1:length(gestureNames)
                gName = gestureNames{g};
                matFile = fullfile(path, user, strcat(gName, '.mat'));
                if ~isfile(matFile), continue; end
                loaded = load(matFile);
                repsData = loaded.reps.(gName).data; % cell array of structs

                for r = 1:length(repsData)
                    s = repsData{r};
                    emgRaw = s.emg; % already [Nx8] double matrix

                    % Build emg as struct with ch1..ch8
                    emgStruct = struct();
                    for ch = 1:size(emgRaw, 2)
                        emgStruct.(sprintf('ch%d', ch)) = emgRaw(:, ch);
                    end
                    sigLen = size(emgRaw, 1);

                    % Build groundTruth from pointGestureBegins
                    if isequal(gName, 'relax') || isempty(s.pointGestureBegins)
                        gt = zeros(sigLen, 1);
                        gtIdx = [1, sigLen];
                        assignedName = 'noGesture';
                    else
                        startPt = s.pointGestureBegins;
                        gt = zeros(sigLen, 1);
                        gt(startPt:end) = 1;
                        gtIdx = [startPt, sigLen];
                        assignedName = gName;
                    end

                    key = sprintf('idx_%d', idx);
                    allSamples.(key).emg              = emgStruct;
                    allSamples.(key).gestureName      = assignedName;
                    allSamples.(key).groundTruth      = gt;
                    allSamples.(key).groundTruthIndex = gtIdx;
                    idx = idx + 1;
                end
            end

            % Split 80/20 training / validation
            keys = fieldnames(allSamples);
            n = length(keys);
            rng(9);
            perm = randperm(n);
            splitIdx = floor(0.8 * n);

            trainingSamples   = struct();
            validationSamples = struct();
            for k = 1:splitIdx
                trainingSamples.(keys{perm(k)}) = allSamples.(keys{perm(k)});
            end
            for k = splitIdx+1:n
                validationSamples.(keys{perm(k)}) = allSamples.(keys{perm(k)});
            end
        end

        % FUNCTION TO READ A FILE
        function data = readFile(filename)
            % Load a Matlab file
            data = load(filename).data;
        end

        % FUNCTION TO GET THE EMG SIGNAL
        function signal = getSignal(emg)
            % Get chanels
            channels = fieldnames(emg);
            % Signal dimensions (1000 x 8 aprox)
            signal = zeros(length(emg.(channels{1})), length(channels));
            for j = 1:length(channels)
                signal(:,j) = emg.(channels{j});
            end
        end

        % FUNCTION TO RECTIFY EMG
        function rectifiedEMG = rectifyEMG(rawEMG, rectFcn)
            switch rectFcn
                case 'square'
                    rectifiedEMG = rawEMG.^2;
                case 'abs'
                    rectifiedEMG = abs(rawEMG);
                case 'none'
                    rectifiedEMG = rawEMG;
                otherwise
                    fprintf('Wrong rectification function. Valid options are square, abs and none');
            end
        end

        % FUNCTION TO APLY A FILTER TO EMG
        function EMGsegment_out = preProcessEMGSegment(EMGsegment_in, Fa, Fb, rectFcn)
            % Normalization
            if max( abs(EMGsegment_in(:)) ) > 1
                drawnow;
                EMGnormalized = EMGsegment_in/128;
            else
                EMGnormalized = EMGsegment_in;
            end
            EMGrectified = Shared.rectifyEMG(EMGnormalized, rectFcn);
            EMGsegment_out = filtfilt(Fb, Fa, EMGrectified);
        end

        % FUNCTION TO PREPROCESS A SIGNAL
        function signal = preprocessSignal(signal)
            [Fb, Fa] = butter(5, 0.1, 'low');
            signal = Shared.preProcessEMGSegment(signal, Fa, Fb, 'abs');
        end

        % FUNCTION TO GENERATE SPECTROGRAMS
        function spectrograms = generateSpectrograms(signal)
            % Spectrogram parameters
            sampleFrecuency = 200;
            % Preallocate space for the spectrograms
            numCols = floor((length(signal) - Shared.OVERLAPPING) / ...
                (Shared.WINDOW - Shared.OVERLAPPING));
            spectrograms = zeros(length(Shared.FRECUENCIES), numCols, Shared.numChannels);
            % Spectrograms generation
            for i = 1:size(signal, 2)
                [s,~,~,~] = spectrogram(signal(:,i), Shared.WINDOW, Shared.OVERLAPPING, ...
                    Shared.FRECUENCIES, sampleFrecuency, 'yaxis'); % [~,~,~,ps]
                spectrograms(:,:,i) = abs(s).^2; % ps;
            end
        end

        % FUNCTION TO SHUFFLE SAMPLES IN A FILE DATASTORE
        function [fds, idx] = shuffle(fds)
            % Get the number of files
            numObservations = numel(fds.Files);
            % Shuffle files and their corresponding labels
            rng(9); % seed
            idx = randperm(numObservations);
            fds.Files = fds.Files(idx);
        end

        % FUCNTION TO SET THE USE OF NOGESTURE
        function classes = setNoGestureUse(withNoGesture)
            if withNoGesture
                classes = ["fist", "noGesture", "open", "pinch", "waveIn", "waveOut"];
            else
                classes = ["fist", "open", "pinch", "waveIn", "waveOut"];
            end
        end

        % FUNCTION TO CREATE LABELS
        function [labels, numObservations] = createLabels(files, withNoGesture)
            % Get the number of files
            numObservations = numel(files);
            % Allocate spacce for labels
            labels = cell(numObservations,1);
            parfor i = 1:numObservations
                file = files{i};
                filepath = fileparts(file); % ../datastore/class
                % The last part of the path is the label
                [~,label] = fileparts(filepath); % [../datastore, class]
                labels{i,1} = label;
            end
            classes = Shared.setNoGestureUse(withNoGesture);
            labels = categorical(labels, classes);
        end

        % FUNCTION TO CLASSIFY PREDICTIONS
        function class = classifyPredictions(yPred)
            categories = Shared.setNoGestureUse(true); % Provemos a quitar el categorical

            % Delete noGestures
            idxs = cellfun(@(label) ~isequal(label,'noGesture'), yPred);
            yPred = categorical(yPred(idxs),Shared.setNoGestureUse(true));

            % Count the number of labels per gesture
            catCounts = countcats(yPred);
            [catCounts,indexes] = sort(catCounts,'descend');
            newCategories = categories(indexes);

            % Set the class if labels are over the theashold
            if catCounts(1) >= Shared.MIN_LABELS_SEQUENCE
                class = newCategories(1);
            else
                class = categorical({'noGesture'}, Shared.setNoGestureUse(true));
            end
        end

        % FUNCTION TO POST PROCESS THE SAMPLE
        function labels = postprocessSample(labels, class)
            if Shared.doPostprocessing
                if ismember(Shared.POSTPROCESS, {'1-1', '2-1', '1-2'})

                    % Check the first label
                    right = isequal(labels{1,1}, 'noGesture');
                    if isequal(Shared.POSTPROCESS, '1-2')
                        right = isequal(labels{1,2}, class) || isequal(labels{1,3}, 'noGesture');
                    end
                    current = isequal(labels{1,1}, class);
                    if right && current
                        labels{1,1} = 'noGesture';
                    end

                    % Set start and finish for middle labels
                    start = 2; finish = length(labels) - 1; % 1-1 by default
                    if isequal(Shared.POSTPROCESS, '2-1')
                        start = 3;
                    elseif isequal(Shared.POSTPROCESS, '1-2')
                        finish = length(labels) - 2;
                    end

                    % Check for misclassified labels
                    for i = start:finish

                        % Check left-current-right classes
                        left = isequal(labels{1,i-1}, class);
                        right = isequal(labels{1,i+1}, class);
                        if isequal(Shared.POSTPROCESS, '2-1')
                            left = isequal(labels{1,i-1}, class) || isequal(labels{1,i-2}, class);
                        elseif isequal(Shared.POSTPROCESS, '1-2')
                            right = isequal(labels{1,i+1}, class) || isequal(labels{1,i+2}, class);
                        end
                        current = ~isequal(labels{1,i}, class);

                        % Replace the class if matches the criterium
                        if left && right && current
                            labels{1,i} = class;
                        end

                        % Replace the class if matches the criterium
                        if ~left && ~right && ~current
                            labels{1,i} = 'noGesture';
                        end

                    end

                    % Check the last label
                    left = isequal(labels{1,length(labels) - 1}, 'noGesture');
                    if isequal(Shared.POSTPROCESS, '2-1')
                        left = isequal(labels{1, length(labels) - 1}, 'noGesture') || ...
                            isequal(labels{1, length(labels) - 2}, 'noGesture');
                    end
                    current = isequal(labels{1,length(labels)}, class);

                    % Replace the class if matches the criterium
                    if left && current
                        labels{1,i} = 'noGesture';
                    end



                    % Set wrong labels to noGestute
                    for i = 1:length(labels)
                        if ~isequal(labels{1,i}, class)
                            labels{1,i} = 'noGesture';
                        end
                    end
                end
            end
            % Transform to categorical
            labels = categorical(labels, Shared.setNoGestureUse(true));
        end

    end

end
