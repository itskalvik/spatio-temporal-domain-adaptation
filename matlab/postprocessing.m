S_new_all = S_new_all_backup;
time_length = size(S_new_all, 2);
   
%%%% normalization %%%%%
   for j=1:time_length
     %  S_target_now = S_target(:,j);
       energy_level = sum(S_new_all(:,j));
       %energy_level = max(S_new_all(:,j));
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

colormap(jet)
imagesc(t,f_vec,20*log10(S_new_all))
set(gca,'YDir','normal')
xlabel('Time (s)')
ylabel('Doppler (Hz)')
title('Micro-Doppler Signature (150MHz, 512FFT)')
axis xy 
colorbar