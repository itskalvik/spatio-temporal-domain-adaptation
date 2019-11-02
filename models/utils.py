import os
import h5py
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from sklearn.model_selection import train_test_split

'''
Resizes images in batches, only 2nd and 3rd dimensions will be resized
args:
    data: 4D numpy array, with shape [number_samples, height, width, channels]
    output_shape: tuple, with new shape for 2nd and 3rd dimensions
output:
    data: 4D numpy array, with shape [number_samples, new_height, new_width, channels]
'''
def resize_data(data, output_shape=(256, 256)):
  _, height, width, channels = data.shape
  data = data.transpose((1, 2, 3, 0))
  data = resize(data.reshape(height, width, -1), output_shape)
  data = data.reshape(*output_shape, channels, -1)
  data = data.transpose((3, 0, 1, 2))
  return data


'''
Returns data, labels and classes from h5py file
args:
    filename: string, filename of h5py dataset
output:
    data: tuple, with (X_data, y_data, classes)
          where X_data and y_data are numpy arrays and classes is a list
'''
def get_h5dataset(filename):
  hf = h5py.File(filename, 'r')
  X_data = np.expand_dims(hf.get('X_data'), axis=-1)
  y_data = np.array(hf.get('y_data'))
  classes = list(hf.get('classes'))
  classes = [n.decode("ascii", "ignore") for n in classes]
  hf.close()
  return X_data, y_data, classes


'''
Balances the dataset to have same number of samples in every class and every day
args:
    X_data: numpy array, feature data [number_samples, ...]
    y_data: numpy array, label data [number_samples, 2], labels have to be sparse
                         with 1st dim for class label and 2nd dim for day
    num_days: int, total number of days in the dataset
    num_classes: int, total number of classes in the dataset
    max_samples_per_class: int, maximum number of samples to keep in each class per day
output:
    data: tuple, with (X_data, y_data, classes)
          where X_data and y_data are numpy arrays and classes is a list
'''
def balance_dataset(X_data, y_data, num_days=10, num_classes=10, max_samples_per_class=95):
  X_data_tmp, y_data_tmp = list(), list()
  for day in range(num_days):
    for idx in range(num_classes):
      X_data_tmp.extend(X_data[(y_data[:, 0] == idx) & (y_data[:, 1] == day)][:max_samples_per_class])
      y_data_tmp.extend(y_data[(y_data[:, 0] == idx) & (y_data[:, 1] == day)][:max_samples_per_class])
  return np.array(X_data_tmp), np.array(y_data_tmp)


'''
mean centers numpy array
args:
    X_data: numpy array, feature data [number_samples, ...]
    data_mean: None or double, mean value used to center data
               if None it is computed from X_data
output:
    data: tuple, with (X_data, data_mean)
          where X_data is a numpy arrays, data_mean is a double
'''
def mean_center(X_data, data_mean=None):
  if data_mean is None:
    data_mean = np.mean(X_data)
  X_data -= data_mean
  return X_data, data_mean


'''
normalizes numpy array to [-1, 1]
args:
    X_data: numpy array, feature data [number_samples, ...]
    data_min: None or double, minimum value used for normalization
              if None it is computed from X_data
    data_ptp: None or double, ptp value used for normalization
              if None it is computed from X_data
output:
    data: tuple, with (X_data, data_min, data_ptp)
          where X_data is a numpy arrays, data_min and data_ptp
          are doubles
'''
def normalize(X_data, data_min=None, data_ptp=None):
  if (data_ptp is None) or (data_min is None):
    data_min = np.min(X_data)
    data_ptp = np.ptp(X_data)
  X_data = 2.*(X_data - data_min)/data_ptp-1
  return X_data, data_min, data_ptp


'''
preprocess target domain data
args:
    filename: string, filename of h5py dataset
    src_classes: list, class names from source domain
    train_trg_days: number of days to use as training data
output:
    X_train_trg: processed training features
    y_train_trg: processed training labels
    X_test_trg: processed testing features
    y_test_trg: processed testing labels
'''
def get_trg_data(filename, src_classes, train_trg_days):
  X_data_trg, y_data_trg, trg_classes = get_h5dataset(filename)
  X_data_trg = resize_data(X_data_trg)

  #split days of data to train and test
  X_train_trg = X_data_trg[y_data_trg[:, 1] < train_trg_days]
  y_train_trg = y_data_trg[y_data_trg[:, 1] < train_trg_days, 0]
  y_train_trg = np.array([src_classes.index(trg_classes[y_train_trg[i]]) for i in range(y_train_trg.shape[0])])

  X_test_trg = X_data_trg[y_data_trg[:, 1] >= train_trg_days]
  y_test_trg = y_data_trg[y_data_trg[:, 1] >= train_trg_days, 0]
  y_test_trg = np.eye(len(src_classes))[y_test_trg]

  if(X_train_trg.shape[0] != 0):
    X_train_trg, trg_mean = mean_center(X_train_trg)
    X_train_trg, trg_min, trg_ptp = normalize(X_train_trg)
    y_train_trg = np.eye(len(src_classes))[y_train_trg]

    X_test_trg, _    = mean_center(X_test_trg, trg_mean)
    X_test_trg, _, _ = normalize(X_test_trg, trg_min, trg_ptp)
  else:
    X_test_trg, _    = mean_center(X_test_trg)
    X_test_trg, _, _ = normalize(X_test_trg)

  X_train_trg = X_train_trg.astype(np.float32)
  y_train_trg = y_train_trg.astype(np.uint8)
  X_test_trg  = X_test_trg.astype(np.float32)
  y_test_trg  = y_test_trg.astype(np.uint8)

  return X_train_trg, y_train_trg, X_test_trg, y_test_trg
