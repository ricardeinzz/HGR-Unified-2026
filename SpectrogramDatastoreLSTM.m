classdef SpectrogramDatastoreLSTM < matlab.io.Datastore & matlab.io.datastore.MiniBatchable
    % SpectrogramDatastoreLSTM specifically for sequence data (CNN-LSTM).
    
    properties
        FilePaths
        CurrentFileIndex
        BatchSize = 1;
    end
    
    properties (Dependent)
        FrameDimensions
        NumClasses
        NumObservations
    end
    
    methods
        function ds = SpectrogramDatastoreLSTM(folderPath)
            % Get file list
            filePattern = fullfile(folderPath, '*.mat');
            ds.FilePaths = dir(filePattern);
            ds.FilePaths = fullfile({ds.FilePaths.folder}, {ds.FilePaths.name})';
            ds.CurrentFileIndex = 1;
            ds.MiniBatchSize = 1;
        end
        
        function [data, info] = read(ds)
            % Read one sequence (MiniBatchSize usually 1 for LSTM in this project)
            fileData = load(ds.FilePaths{ds.CurrentFileIndex});
            
            % Expected variables: sequences, labelsSequences
            data = struct();
            data.sequences = fileData.sequences;
            data.labelsSequences = fileData.labelsSequences;
            
            ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            info = struct();
        end
        
        function tf = hasdata(ds)
            tf = ds.CurrentFileIndex <= length(ds.FilePaths);
        end
        
        function reset(ds)
            ds.CurrentFileIndex = 1;
        end
        
        function dims = get.FrameDimensions(ds)
            if ~isempty(ds.FilePaths)
                sample = load(ds.FilePaths{1}).sequences;
                dims = [size(sample{1}, 1), size(sample{1}, 2), size(sample{1}, 3)];
            else
                dims = [63, 6, 7];
            end
        end
        
        function n = get.NumClasses(ds)
            n = 6; % Fixed for this project
        end
        
        function n = get.NumObservations(ds)
            n = length(ds.FilePaths);
        end
        
        function dsNew = setDataAmount(ds, amount)
            numSamples = floor(length(ds.FilePaths) * amount);
            indices = randperm(length(ds.FilePaths), numSamples);
            ds.FilePaths = ds.FilePaths(indices);
            dsNew = ds;
        end
    end
end
