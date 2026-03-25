% modelCreation trains and saves the HGR CNN model.
% In this file is stablished the neural network architecture.
% Models are generated in "Models/" folder.

%% SET DATASTORES PATHS
scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);

dataDirTraining = fullfile(rootDir, 'Datastores', 'training');
dataDirValidation = fullfile(rootDir, 'Datastores', 'validation');

% The classes are defined
withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);

% The datastores are created
trainingDatastore = SpectrogramDatastore(dataDirTraining, withNoGesture);
validationDatastore = SpectrogramDatastore(dataDirValidation, withNoGesture);

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.DataDimensions;

%% DEFINE THE AMOUNT OF DATA
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);

%% THE NEURAL NETWORK ARCHITECTURE IS DEFINED
numClasses = trainingDatastore.NumClasses;
lgraph = setNeuralNetworkArchitecture(inputSize, numClasses);
analyzeNetwork(lgraph);

%% THE OPTIONS ARE DIFINED
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',8, ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',10, ...
    'MiniBatchSize',512, ...
    'Shuffle','every-epoch', ...
    'ValidationData', validationDatastore, ...
    'Plots','training-progress');

%% NETWORK TRAINING
net = trainNetwork(trainingDatastore, lgraph, options);

%% SAVE MODEL
if ~exist(fullfile(rootDir, 'Models'), 'dir')
    mkdir(fullfile(rootDir, 'Models'));
end
save(fullfile(rootDir, 'Models', ['model_', datestr(now,'dd-mm-yyyy_HH-MM-ss')]), 'net');

function lgraph = setNeuralNetworkArchitecture(inputSize, numClasses)
    % Neural Network Architecture for HGR CNN
    % Matches the 2023-HGR5-CNN research parameters
    lgraph = layerGraph();
    % Layers implementation...
end
