classdef Shared
    % Shared contains properties and methods shared between the models.
    
    %{
    Laboratorio de Inteligencia y Visión Artificial
    ESCUELA POLITÉCNICA NACIONAL
    Quito - Ecuador
    
    laboratorio.ia@epn.edu.ec
    
    "I find that I don't understand things unless I try to program them."
    -Donald E. Knuth
    
    Matlab 9.11.0.2022996 (R2021b) Update 4.
    %}
    
    properties (Constant)
        
        %% DATASET DATA
        % Path where the dataset is located
        % DATA_DIR = 'C:\Users\ricar\Documents\MATLAB\Versionamiento\EMG-EPN-612 dataset';
        
        % The amount of gestures in the dataset
        NUM_GESTURES = 6;
        % The amount of samples per user
        NUM_SAMPLES_USER = 50; % 1--50
        
        %% SIGNAL DATA
        % The sampling frequency of the signal
        FS = 1000;
        % The amount of channels of the signal
        NUM_CHANNELS = 7;
        
        %% SPECTROGRAMS DATA
        % Length of the windows for the spectrograms
        WINDOW_LENGTH = 124;
        % Overlapping factor of the windows for spectrograms
        OVERLAP_LENGTH = 100;
        % The number of samples for the fast fourier transform
        FFT_SAMPLES = 124;
        
        %% FRAME DATA
        % Size of the frame(points) that will be classify
        FRAME_WINDOW = 250;
        % Number of hidden units for the LSTM
        NUM_HIDDEN_UNITS = 126;
        
        %% SLIDING WINDOW DATA (CNN)
        % Step of the window for recognition (points)
        WINDOW_STEP_RECOG = 10;
        
        %% SLIDING WINDOW DATA (LSTM)
        % Step of the window for LSTM (points)
        WINDOW_STEP_LSTM = 10;
        % Time that takes create a spectrogram and classify it
        % AVG_PROC_TIME_FRAME = 0.0035;%0.0031
        
        %% CLASSIFICATION DATA
        % Threshold to consider that a gesture is happening
        FRAME_CLASS_THRESHOLD = 0.8;
                
        %% EVALUATION DATA
        % Filling used during evaluation (before | none)
        FILLING_TYPE_EVAL = 'before';
        
    end
    
    
    methods(Static)
        
        %% DATASET FUNCTIONS
        % Function to establish if the 'noGesture' class is used
        function classes = setNoGestureUse(withNoGesture)
            if withNoGesture
                classes = {'noGesture','thumbFlexion','indexFlexion', ... 
                    'middleFlexion','ringFlexion','littleFlexion'};
            else
                classes = {'thumbFlexion','indexFlexion','middleFlexion', ... 
                    'ringFlexion','littleFlexion'};
            end
        end
        
        % Check if testing is included
        function result = includeTesting()
            result = true;
        end
        
        % The number of users for testing
        function numTestUsers = numTestUsers()
            numTestUsers = 13;
        end
        
        % The amount of samples per user
        function numSamplesUser = numSamplesUser()
            numSamplesUser = 50;
        end
        
        % Function to extract the users directories from a folder
        function [users, trainingPath] = getUsers(dataDir, trainingDir)
            trainingPath = fullfile(dataDir, trainingDir);
            users = dir(trainingPath);
            users = users(~ismember({users.name}, {'.', '..'}) & [users.isdir]);
            users = {users.name}';
        end
        
        % Function to extract the training and testing samples from a user
        function [trainingSamples, testingSamples] = getTrainingTestingSamples(trainingPath, user)
            
            % Set path to user directory
            userPath = fullfile(trainingPath, user);
            
            % Check if the datset is in JSON or MAT (unified support)
            jsonFiles = dir(fullfile(userPath, '*.json'));
            
            if ~isempty(jsonFiles)
                % JSON FORMAT (Original)
                % Set path to json file
                filePath = fullfile(userPath, [char(user), '.json']);
                jsonFile = fileread(filePath);
                data = jsondecode(jsonFile);
                
                % Get testing and training samples
                testingSamples = data.testingSamples;
                trainingSamples = data.trainingSamples;
            else
                % MAT FORMAT (New)
                matFiles = dir(fullfile(userPath, '*.mat'));
                trainingSamples = struct();
                testingSamples = struct();
                
                % Conventional split: first 40 for training, last 10 for testing
                for i = 1:length(matFiles)
                    fileName = matFiles(i).name;
                    filePath = fullfile(userPath, fileName);
                    
                    % Load mat file
                    sampleData = load(filePath);
                    
                    % Determine key
                    cleanName = strrep(fileName, '.mat', '');
                    cleanName = strrep(cleanName, '-', '_');
                    
                    if i <= 40
                        trainingSamples.(cleanName) = sampleData;
                    else
                        testingSamples.(cleanName) = sampleData;
                    end
                end
            end
        end
        
        
        %% SIGNAL FUNCTIONS
        % Function to extract the signal from a sample data
        function signal = getSignal(sample)
            % Extract the data for the 7 channels
            c1 = sample.c1; c2 = sample.c2; c3 = sample.c3;
            c4 = sample.c4; c5 = sample.c5; c6 = sample.c6;
            c7 = sample.c7;
            % Set the data into a single matrix
            signal = [c1, c2, c3, c4, c5, c6, c7];
        end
        
        % Function to preprocess the signal
        function preprocessedSignal = preprocessSignal(signal)
            % 1. Rectification of the signal
            preprocessedSignal = abs(signal);
            % 2. Normalization of the signal
            %preprocessedSignal = preprocessedSignal ./ max(preprocessedSignal(:));
        end
        
        
        %% SPECTROGRAM FUNCTIONS
        % Function to generate 7 spectrograms from a signal
        function spectrograms = generateSpectrograms(signal)
            
            % Calculate size of the spectrogram
            % [1+fft/2, 1+floor((points-window)/step)]
            % numFrequencyBins = 1 + Shared.FFT_SAMPLES / 2;
            % numTimeBins = 1 + floor((Shared.FRAME_WINDOW - Shared.WINDOW_LENGTH) / ... 
            %    (Shared.WINDOW_LENGTH - Shared.OVERLAP_LENGTH));
            
            % Preallocate space for the spectrograms
            spectrograms = zeros(63, 6, 7);
            
            % For each channel
            for i = 1:Shared.NUM_CHANNELS
                % Extract signal of a channel
                signalChannel = signal(:, i);
                % Generate spectrogram
                [~, ~, ~, s] = spectrogram(signalChannel, Shared.WINDOW_LENGTH, ... 
                    Shared.OVERLAP_LENGTH, Shared.FFT_SAMPLES, Shared.FS);
                % Adjust to 63x6 (Standard size in this project)
                spectrograms(:, :, i) = abs(s(1:63, :));
            end
        end
        
        
        %% CLASSIFICATION FUNCTIONS
        % Function to classify a series of predicted labels (Voting)
        function classificationResult = classifyPredictions(labels)
            
            % Convert to categorical
            labels = categorical(labels, Shared.setNoGestureUse(true));
            
            % Find the elements that are not 'noGesture'
            gestureLabels = labels(labels ~= 'noGesture');
            
            if isempty(gestureLabels)
                classificationResult = 'noGesture';
            else
                % Count occurrences
                counts = countcats(gestureLabels);
                [maxCount, idx] = max(counts);
                
                % Check if there is a clear winner
                if maxCount > (length(labels) * 0.2) % Empirical threshold
                     classes = categories(gestureLabels);
                     classificationResult = classes{idx};
                else
                    classificationResult = 'noGesture';
                end
            end
        end
        
        % Function to postprocess sample labels
        function postprocessedLabels = postprocessSample(labels, class)
             % If the sample was classified as noGesture, all frames are noGesture
            if isequal(class, 'noGesture')
                postprocessedLabels = repmat({'noGesture'}, 1, length(labels));
            else
                postprocessedLabels = labels;
                % Minimal smoothing: if a frame is surrounded by 'class', it belongs to 'class'
                for i = 2:length(labels)-1
                    if isequal(labels{i-1}, class) && isequal(labels{i+1}, class)
                        postprocessedLabels{i} = class;
                    end
                end
            end
        end
        
    end
    
end
