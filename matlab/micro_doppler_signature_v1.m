clc;
clear;
close all;

iqmat = readDCA1000_1642("/mnt/archive3/pose/kjakkala/10_14/17_30_08/adc_data_Raw_6.bin");

fc = 77e9;
c = 3e8;
lambda = c/fc;
BW = 900.9*1e6;
Nsweep = 230;  %number of chirps per frame
ADC_num = 256; %number of ADC samples per chirp
frame_period = 33.0*1e-3; % frame periodility  
max_range = 5; % maximum range
min_range = 0.1; % minimum range
disp(['Bandwidth ', num2str(BW), ' range:', num2str(min_range), ' to ', num2str(max_range), ' meter'])

S_new_all = 0; % we will aggregate the spectorgram of four antennas

for n =1:1  %compute spectrogram of each antenna
    disp(['Calculating micro-doppler for rx antenna ', num2str(n)])
    r1 = iqmat(n,:);
    total_data = size(iqmat, 2); %total data we collected
    num_frame = floor(total_data/(Nsweep*ADC_num)); %number of frames
    num_frame_start = 50; %starting number of frames
    num_frame_end = num_frame; %starting number of frames
    num_frame_test = num_frame_end - num_frame_start + 1;
    sampling_f = 1/(frame_period/Nsweep); % the real chrip transmitting frequency
    chirp_duration = frame_period/Nsweep; % the estimated chirp cycle 
    range_res = c/(2*BW); % calculate range resolution
    vel_res = lambda/(2*frame_period); % calculate velocity resolution

    % Re-arrange them for frames
    r1 = r1(1:end, 1:ADC_num*Nsweep*num_frame);
    mat = reshape(r1, [], num_frame);

    new_waveform = [];
    for n = num_frame_start:num_frame_end
        current_frame = mat(:,n); % select which frame to analyze 
        current_xr = reshape(current_frame, ADC_num, []); 
        new_waveform = [new_waveform current_xr];
    end

    window = hann(ADC_num); % use hanning window
    range_matrix = fft(new_waveform.*window); % use windowing to remove frequency leakage

    %%%% translate range into range bin index
    range_min = ceil(min_range/range_res);
    range_max = ceil(max_range/range_res);

    for i =range_min:range_max
        current_sig = range_matrix(i,:);
        current_sig = highpass(current_sig,50,sampling_f);
        range_matrix(i,:) = current_sig;
    end

    y = range_min:range_max; % show range bin index 
    x =  chirp_duration*(Nsweep*num_frame_start:Nsweep*num_frame_end);
    range_matrix_abs = abs(range_matrix([range_min:range_max],:));
    max_range_matrix = max(range_matrix_abs);
    range_matrix_dbfs = 20*log10(range_matrix_abs./max_range_matrix);
    clims = [-30 0];

    %%%%% plot range-time figure %%%%%%
    figure;
    imagesc(x,y,range_matrix_dbfs,clims)
    xlabel('Time (s)')
    ylabel('range index')
    title('range-time map')

    %%%% compute micro-doppler signatures%%%%%%%%
    S_new =0;
    NFFT = 512; 
    WINDOW = chebwin(NFFT,120);
    n = 1;
    NOVERLAP = NFFT - n;
    Fs = sampling_f;

    for i=range_min:range_max
        disp(['Calculating spectrogram for range bin ',num2str(i),'/',num2str(range_max)])
        f_vec = [-floor(NFFT/2) : ceil(NFFT/2)-1] * Fs/NFFT;
        [S,f_vec,t] = spectrogram(range_matrix(i,:),WINDOW,NOVERLAP,f_vec,Fs,'yaxis');
        S_new = abs(S) + S_new;
    end

    S_new_all = S_new + S_new_all;
end
S_new_all_backup = S_new_all;
S_all_max = max(max(S_new_all));
S_new_all = S_new_all_backup;
time_length = size(S_new_all, 2);
   
%%%% normalization %%%%%
for j=1:time_length
   energy_level = sum(S_new_all(:,j));
   S_new_all(:,j) = S_new_all(:,j)/energy_level;
end

%%%% frequency-domain denoise %%%% 
mean_target = mean(S_new_all(:));
S_new_all = S_new_all - mean_target;
index = S_new_all<0;
if sum(index(:))~=0
S_new_all(S_new_all<0)=0;
end

%%%%%% 2D Guassian filter %%%%%%
hsize = 15;
sigma = 2;
h = fspecial('gaussian',hsize,sigma);
S_new_all = imfilter(S_new_all,h,'replicate');
S_max = max(max(S_new));

%S_new_all = flipud(S_new_all);
figure;
colormap(jet)
imagesc(t, f_vec,20*log10(S_new_all))
set(gca,'YDir','normal')
xlabel('Time (s)')
ylabel('Doppler (Hz)')
title('Micro-Doppler Signature (150MHz, 512FFT)')
set(gca,'clim',[-120 0]); %900MHz
axis xy 
colorbar