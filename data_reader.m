%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dater_reader: the function for reading fraud data in csv format %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [data] = data_reader(data_path, data_format, year_start, year_end)
temp = csvread(data_path, 1, 0);
switch data_format
    case 'data_default' % read data with 28 raw accounting variables
        data.years = temp(:, 1);
        idx = data.years>=year_start & data.years<=year_end;
        data.years = temp(idx, 1);
        data.firms = temp(idx, 2);
        data.paaers = temp(idx, 3);
        data.labels = temp(idx, 4);
        data.features = temp(idx,5:32);
        data.num_obervations = size(data.features,1);
        data.num_features = size(data.features,2);
    otherwise
        disp('Error: unsupported data format!');
end

fprintf('Data Loaded: %s, %d features, %d observations.\n',data_path, data.num_features, data.num_obervations);

end
