classdef SpectrogramDatastore < matlab.io.Datastore & matlab.io.datastore.MiniBatchable
    % SpectrogramDatastore custom datastore for HGR spectrograms.
    
    properties
        FilePaths
        Labels
        CurrentFileIndex
        WithNoGesture
    end
    
    properties (Dependent)
        DataDimensions
        NumClasses
    end
    
    methods
        function ds = SpectrogramDatastore(folderPath, withNoGesture)
            % Initialize properties
            ds.WithNoGesture = withNoGesture;
            
            % Get file list
            filePattern = fullfile(folderPath, '**/*.mat');
            ds.FilePaths = dir(filePattern);
            ds.FilePaths = fullfile({ds.FilePaths.folder}, {ds.FilePaths.name})';
            
            % Extract labels from folder names
            ds.Labels = categorical(cell(length(ds.FilePaths), 1), Shared.setNoGestureUse(withNoGesture));
            for i = 1:length(ds.FilePaths)
                [parentPath, ~, ~] = fileparts(ds.FilePaths{i});
                [~, labelName, ~] = fileparts(parentPath);
                ds.Labels(i) = categorical({labelName});
            end
            
            ds.CurrentFileIndex = 1;
            ds.MiniBatchSize = 128;
        end
        
        function [data, info] = read(ds)
            % Read a minibatch
            batchSize = min(ds.MiniBatchSize, length(ds.FilePaths) - ds.CurrentFileIndex + 1);
            
            data = table(cell(batchSize, 1), cell(batchSize, 1));
            
            for i = 1:batchSize
                idx = ds.CurrentFileIndex + i - 1;
                fileData = load(ds.FilePaths{idx}).spectrograms;
                data.Var1{i} = fileData;
                data.Var2{i} = ds.Labels(idx);
            end
            
            ds.CurrentFileIndex = ds.CurrentFileIndex + batchSize;
            info = struct();
        end
        
        function tf = hasdata(ds)
            tf = ds.CurrentFileIndex <= length(ds.FilePaths);
        end
        
        function reset(ds)
            ds.CurrentFileIndex = 1;
        end
        
        function dims = get.DataDimensions(ds)
            if ~isempty(ds.FilePaths)
                sample = load(ds.FilePaths{1}).spectrograms;
                dims = size(sample);
            else
                dims = [63, 6, 7];
            end
        end
        
        function n = get.NumClasses(ds)
            n = length(categories(ds.Labels));
        end
        
        function dsNew = setDataAmount(ds, amount)
            % Keep only a percentage of data
            numSamples = floor(length(ds.FilePaths) * amount);
            indices = randperm(length(ds.FilePaths), numSamples);
            ds.FilePaths = ds.FilePaths(indices);
            ds.Labels = ds.Labels(indices);
            dsNew = ds;
        end
    end
end
