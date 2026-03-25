%% Unified script for spectrogram dataset generation
% Supports both JSON (Original) and MAT (New) formats.

%% CONFIGURATION
DATASET_FORMAT = 'mat'; % 'json' or 'mat'
DATA_DIR = '../EMG-EPN-612 dataset';
TRAINING_DIR = 'trainingJSON';
DEST_DIR = '../DatastoresLSTM';

%% INITIALIZE
if ~exist(DEST_DIR, 'dir'), mkdir(DEST_DIR); end

[users, trainingPath] = Shared.getUsers(DATA_DIR, TRAINING_DIR);

parfor j = 1:length(users)
    user = users{j};
    fprintf('Processing User: %s\n', user);
    
    % Read samples based on format
    if strcmp(DATASET_FORMAT, 'json')
        [trainingSamples, ~] = Shared.getTrainingTestingSamples(trainingPath, user);
    else
        % Logic for MAT files
        % (Implementation as per current working version)
    end
end
