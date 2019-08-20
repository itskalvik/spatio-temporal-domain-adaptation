function [retVal] = readDCA1000_1443(fileName)

%% global variables
% change based on sensor config
numADCBits = 16; % number of ADC bits per sample
numLanes = 4; % do not change. number of lanes is always 4 even if only 1 lane is used. unused lanes
isReal = 0; % set to 1 if real only data, 0 if complex dataare populated with 0 %% read file and convert to signed number



% read .bin file
fid = fopen(fileName,'r');
adcData = fread(fid, 'int16');

% if 12 or 14 bits ADC per sample compensate for sign extension
if numADCBits ~= 16
    l_max = 2^(numADCBits-1)-1;
    adcData(adcData > l_max) = adcData(adcData > l_max) - 2^numADCBits;
end
fclose(fid);

%% organize data by LVDS lane
% for real only data
if isReal
    % reshape data based on one samples per LVDS lane
    adcData = reshape(adcData, numLanes, []);
%for complex data
else
    % reshape and combine real and imaginary parts of complex number
    adcData = reshape(adcData, numLanes*2, []);
    adcData = adcData([1,2,3,4],:) + sqrt(-1)*adcData([5,6,7,8],:);
end
%% return receiver data
retVal = adcData;
