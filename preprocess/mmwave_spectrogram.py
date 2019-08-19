from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from joblib import Parallel, delayed 
from skimage import transform
from scipy.io import loadmat
from scipy import ndimage
from scipy import signal
import numpy as np
import argparse
import pickle
import time
import h5py
import math
import sys
import os

def read_samples(dataset_path, endswith=".bin", num_keep=100):
  datapaths, labels = list(), list()
  classes = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and not f.startswith('.')])

  for c in classes:
    c_dir = os.path.join(dataset_path, c)
    dates = sorted(os.listdir(c_dir))
    for date in dates:
      samples = os.listdir(os.path.join(c_dir, date))
      for sample in samples:
        if sample.endswith(endswith):
          datapaths.append(os.path.join(c_dir, date, sample))
          labels.append([classes.index(c), dates.index(date)])
  return datapaths, labels, classes

def fspecial_gaussian(size=15, sigma=2):
    kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(size-1)/2)**2+(y-(size-1)/2)**2))/(2*sigma**2)), (size, size))
    kernel /= np.sum(kernel)
    return kernel

def smooth(x,window_len):
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    w=np.hanning(window_len)
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len//2):-(window_len//2)]

def cfar(x, num_train, num_guard, rate_fa):
    num_cells = x.size
    num_train_half = round(num_train / 2)
    num_guard_half = round(num_guard / 2)
    num_side = num_train_half + num_guard_half
 
    alpha = num_train*(rate_fa**(-1/num_train) - 1)
    
    peak_idx = []
    for i in range(num_side, num_cells - num_side):
        
        if i != i-num_side+np.argmax(x[i-num_side:i+num_side+1]): 
            continue
        
        sum1 = np.sum(x[i-num_side:i+num_side+1])
        sum2 = np.sum(x[i-num_guard_half:i+num_guard_half+1]) 
        p_noise = (sum1 - sum2) / num_train 
        threshold = alpha * p_noise
        
        if x[i] > threshold: 
            peak_idx.append(i)
    
    peak_idx = np.array(peak_idx, dtype=int)
    
    return peak_idx

def detect_peak(data):
    data = np.abs(data[250:])

    for _ in range(6):
        data -= np.mean(data)
        data[np.where(data < 0)] = 0
    data = smooth(data, 2001)

    if(np.var(data) < 5000):
        return False, [] 
    
    peak_idx = cfar(data, num_train=200, num_guard=50, rate_fa=1)
    k_inds = peak_idx[np.where(data[peak_idx] > (np.max(data)/3))]
    k_inds.sort()
    k_inds+=250
    
    if (len(k_inds) > 0):
        return True, k_inds
    else:
        return False, []

def get_spectrogram(fname, label):
    slices = []
    iq_data = loadmat(fname)["iqmat"]
    
    data = iq_data[0,:].reshape((-1, num_frame), order="F")

    tmp = []
    for j in range(num_frame):
        tmp.append(data[:, j].reshape((num_adc, num_chirp), order="F"))
    data = np.hstack(tmp)
    
    data = (data.transpose()*signal.hann(num_adc)).transpose()
    range_matrix = np.fft.fft(data, axis=0)
        
    b, a = signal.butter(8, 50/(fs/2), 'high')
    start = False
    frame_slices = []
    start_id = range_min
    end_id = range_max
    direc = None
    for j in range(range_min, range_max):
        range_matrix[j, :] = signal.lfilter(b, a, range_matrix[j,:], axis=0)    
        tmp_status, peak_ids = detect_peak(range_matrix[j, :])
        
        if (start and not tmp_status):
            end_id = j
            break
        elif tmp_status and not start:
            start = True
            start_id = j
            if (peak_ids[-1] > range_matrix[j, :].shape[-1]/2):
                direc = 0
            else:
                direc = -1
            init_window = peak_ids[direc]
            frame_slices.append(init_window)
                        
            range_matrix[j, :init_window-attention_window_length] = 0
            range_matrix[j, init_window+attention_window_length:] = 0
        elif start:            
            if (direc == 0):
                init_window = peak_ids[np.where(np.logical_and(peak_ids >= init_window-4*attention_window_length, peak_ids <= (init_window+attention_window_length)))]
            else:
                init_window = peak_ids[np.where(np.logical_and(peak_ids <= init_window+4*attention_window_length, peak_ids >= (init_window-attention_window_length)))]         
            
            if len(init_window) == 0:
                end_id = j
                break
                
            init_window = init_window[direc]
            frame_slices.append(init_window)
            
            range_matrix[j, :init_window-attention_window_length] = 0
            range_matrix[j, init_window+attention_window_length:] = 0
            
    if(len(frame_slices) < 1):
        return np.array([]), label

    range_matrix = range_matrix[start_id:end_id]    
    for j in range(range_matrix.shape[0]):
        f_vec, t, S = signal.spectrogram(range_matrix[j,:], 
                                        fs=fs, 
                                        window=spec_window,
                                        noverlap=noverlap, 
                                        nfft=nfft,
                                        return_onesided=False,
                                        mode="complex")
        
        if (j == 0):
            S_new_all = np.abs(S)
        else:
            S_new_all += np.abs(S)
            
    S_new_all = np.roll(S_new_all, int(S_new_all.shape[0]/2), axis=0)

    sums = np.sum(S_new_all, 0)
    sum_inds = np.where(sums == 0)[0]
    sums[sum_inds] = 1
    fix_arr = np.ones_like(sums)
    fix_arr[sum_inds] = 0

    S_new_all /= sums
    S_new_all *= fix_arr
    S_new_all -= np.mean(S_new_all)
    S_new_all[np.where(S_new_all < 0)] = 0

    S_new_all = ndimage.convolve(S_new_all, fspecial_gaussian(), mode='nearest')
    S_new_all[np.where(S_new_all <= 0)] = 1e-9
    S_new_all = 20*np.log10(S_new_all)

    if direc == 0:
        S_new_all = S_new_all[128:int(S_new_all.shape[0]/2), int((min(frame_slices)-attention_window_length)/16):int((max(frame_slices)+attention_window_length)/16)]
        S_new_all = S_new_all[:, int(S_new_all.shape[1]/2)-512:int(S_new_all.shape[1]/2)+512]
    elif direc == -1:
        S_new_all = S_new_all[int(S_new_all.shape[0]/2):-128, int((min(frame_slices)-attention_window_length)/16):int((max(frame_slices)+attention_window_length)/16)]
        S_new_all = S_new_all[:, int(S_new_all.shape[1]/2)-512:int(S_new_all.shape[1]/2)+512]      
        S_new_all = np.flip(S_new_all, axis=0)
        S_new_all = np.flip(S_new_all, axis=1)
        
    if (S_new_all.shape != (128, 1024)):
        return np.array([]), label
    
    return S_new_all, label      

#-------------------------------------------------------------------#
c = 3e8
max_range = 5
min_range = 1
band_width = 900.9*1e6
range_res = c/(2*band_width)
range_min = int(np.ceil(min_range/range_res))-1
range_max = int(np.ceil(max_range/range_res))-1

num_frame = 200
num_adc = 256
frame_period = 33*1e-3
num_chirp = 230
fs = 1/(frame_period/num_chirp)
chirp_duration = frame_period/num_chirp

nfft = 512
noverlap = nfft - 16
spec_window = signal.windows.chebwin(nfft, 120)
attention_window_length = int(np.ceil(0.2/chirp_duration))
#-------------------------------------------------------------------#

dataset_file = "/home/kjakkala/mmwave/data/mmwave_source.h5"
src_path = "/mnt/archive2/source/"

files, labels, classes = read_samples(src_path, ".mat")
classes = [n.encode("ascii", "ignore") for n in classes]

dset_X, dset_y = zip(*Parallel(n_jobs=-1)(delayed(get_spectrogram)(files[i], labels[i]) for i in range(len(files))))
dset_X = np.array(dset_X)
dset_y = np.array(dset_y)
print(dset_y.shape, dset_X.shape)

delete_inds = []
for ind in range(len(dset_X)):
    if (dset_X[ind].shape != (128, 1024)):
        delete_inds.append(ind)
        print(files[ind])
print(len(delete_inds))
dset_X = np.delete(dset_X, delete_inds, 0)
dset_y = np.delete(dset_y, delete_inds, 0)

print(dset_y.shape, dset_X.shape)
dset_X = np.asarray(dset_X)
dset_y = np.asarray(dset_y)

hf = h5py.File("/home/kjakkala/neuralwave/data/mmwave_spectorgram.h5", 'w')
hf.create_dataset('X_data', data=dset_X)
hf.create_dataset('y_data', data=dset_y)
hf.create_dataset('labels', data=classes)
hf.close()
