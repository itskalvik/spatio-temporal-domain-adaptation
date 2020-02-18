import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
repo_path = os.getenv('MMWAVE_PATH')
import sys
sys.path.append(os.path.join(repo_path, 'models'))
from utils import *
from resnet_amca import ResNet50AMCA
import numpy as np
import argparse
import yaml
import h5py
from scipy.stats import mode

np.set_printoptions(threshold=np.inf)

from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

@tf.function
def predict(images):
  logits, fc1 = model(images, training=False)
  return tf.nn.softmax(logits), fc1

def get_acc_encodings(data, labels):
  acc = tf.keras.metrics.CategoricalAccuracy(name='acc')
  data_list = []
  for i in range(len(data)):
    logits, encodings = predict(np.expand_dims(data[i], axis=0))
    data_list.append(np.array(encodings))
    acc(logits, labels[i])
  return np.squeeze(data_list), float(acc.result())

def generate_kmeans_model(source_x, source_y, target_x, target_y, num_classes=10):
    # Compute source centers for initializing target clusters
    centers = []
    for i in range(num_classes):
        centers.append(source_x[np.where(np.argmax(source_y, axis=1)==i)].mean(axis=0))
    centers = np.array(centers)

    model = KMeans(n_clusters=num_classes, n_init=10).fit(target_x)
    print(model.labels_)
    print(np.argmax(target_y, axis=1))
    return model, [mode(model.labels_[i:i+50]) for i in range(0, labels.shape[0], 50)]

def get_kmeans_acc(model, data, labels, print_labels=False):
    if data.shape[0] > 0:
        pred_labels = model.predict(data)
        labels = np.argmax(labels, axis=1)
        if print_labels:
            print(pred_labels)
        return (pred_labels==labels).mean()
    else:
        return 0.0

def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--train_source_days', type=int, default=3)
    parser.add_argument('--train_source_unlabeled_days', type=int, default=0)
    parser.add_argument('--train_server_days', type=int, default=0)
    parser.add_argument('--train_conference_days', type=int, default=0)
    parser.add_argument('--log_path', default="logs/")
    parser.add_argument('--enc_path', default="data/encodings-server.h5")
    return parser

if __name__=='__main__':
    parser = get_parser()
    arg = parser.parse_args()

    config_filepath = os.path.join(arg.log_path, 'config.yaml')
    with open(config_filepath) as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
        args = argparse.Namespace(**args)

    os.environ['CUDA_VISIBLE_DEVICES']=arg.gpu

    dataset_path    = os.path.join(repo_path, 'data')
    num_classes     = args.num_classes
    train_source_days = arg.train_source_days
    train_server_days = arg.train_server_days
    train_conference_days = arg.train_conference_days
    train_source_unlabeled_days = arg.train_source_unlabeled_days
    num_features    = args.num_features
    activation_fn   = args.activation_fn
    model_filters   = args.model_filters
    ca              = args.ca

    checkpoint_path = os.path.join(arg.log_path, 'checkpoints')
    encodings_file  = os.path.join(repo_path, arg.enc_path)

    '''
    Data Preprocessing
    '''

    X_data, y_data, classes = get_h5dataset(os.path.join(dataset_path, 'source_data.h5'))
    X_data, y_data = balance_dataset(X_data, y_data,
                                     num_days=10,
                                     num_classes=len(classes),
                                     max_samples_per_class=95)

    #split days of data to train and test
    X_src = X_data[y_data[:, 1] < train_source_days]
    y_src = y_data[y_data[:, 1] < train_source_days, 0]
    y_src = np.eye(len(classes))[y_src]
    X_train_src, X_test_src, y_train_src, y_test_src = train_test_split(X_src,
                                                                        y_src,
                                                                        stratify=y_src,
                                                                        test_size=0.10,
                                                                        random_state=42)

    X_trg = X_data[y_data[:, 1] >= train_source_days]
    y_trg = y_data[y_data[:, 1] >= train_source_days]
    X_train_trg = X_trg[y_trg[:, 1] < train_source_days+train_source_unlabeled_days]
    y_train_trg = y_trg[y_trg[:, 1] < train_source_days+train_source_unlabeled_days, 0]
    y_train_trg = np.eye(len(classes))[y_train_trg]

    X_test_trg = X_data[y_data[:, 1] >= train_source_days+train_source_unlabeled_days]
    y_test_trg = y_data[y_data[:, 1] >= train_source_days+train_source_unlabeled_days, 0]
    y_test_trg = np.eye(len(classes))[y_test_trg]

    del X_src, y_src, X_trg, y_trg, X_data, y_data

    #mean center and normalize dataset
    X_train_src, src_mean = mean_center(X_train_src)
    X_train_src, src_min, src_ptp = normalize(X_train_src)

    X_test_src, _    = mean_center(X_test_src, src_mean)
    X_test_src, _, _ = normalize(X_test_src, src_min, src_ptp)

    if(X_train_trg.shape[0] != 0):
      X_train_trg, trg_mean = mean_center(X_train_trg)
      X_train_trg, trg_min, trg_ptp = normalize(X_train_trg)

      X_test_trg, _    = mean_center(X_test_trg, trg_mean)
      X_test_trg, _, _ = normalize(X_test_trg, trg_min, trg_ptp)
    else:
      X_test_trg, _    = mean_center(X_test_trg, src_mean)
      X_test_trg, _, _ = normalize(X_test_trg, src_min, src_ptp)

    X_train_src = X_train_src.astype(np.float32)
    y_train_src = y_train_src.astype(np.uint8)
    X_test_src  = X_test_src.astype(np.float32)
    y_test_src  = y_test_src.astype(np.uint8)
    X_train_trg = X_train_trg.astype(np.float32)
    y_train_trg = y_train_trg.astype(np.uint8)
    X_test_trg  = X_test_trg.astype(np.float32)
    y_test_trg  = y_test_trg.astype(np.uint8)

    X_train_conf,   y_train_conf,   X_test_conf,   y_test_conf   = get_trg_data(os.path.join(dataset_path,
                                                                                             'target_conf_data.h5'),
                                                                                classes,
                                                                                train_conference_days)
    X_train_server, y_train_server, X_test_server, y_test_server = get_trg_data(os.path.join(dataset_path,
                                                                                             'target_server_data.h5'),
                                                                                classes,
                                                                                train_server_days)
    _             , _             , X_data_office, y_data_office = get_trg_data(os.path.join(dataset_path,
                                                                                             'target_office_data.h5'),
                                                                                classes,
                                                                                0)

    print("\nOriginal Data shapes:")
    print("X_train_src:    {:<20} {:<12}".format(str(X_train_src.shape), str(y_train_src.shape)))
    print("X_test_src:     {:<20} {:<12}".format(str(X_test_src.shape), str(y_test_src.shape)))
    print("X_train_trg:    {:<20} {:<12}".format(str(X_train_trg.shape), str(y_train_trg.shape)))
    print("X_test_trg:     {:<20} {:<12}".format(str(X_test_trg.shape), str(y_test_trg.shape)))
    print("X_train_conf:   {:<20} {:<12}".format(str(X_train_conf.shape), str(y_train_conf.shape)))
    print("X_test_conf:    {:<20} {:<12}".format(str(X_test_conf.shape), str(y_test_conf.shape)))
    print("X_train_server: {:<20} {:<12}".format(str(X_train_server.shape), str(y_train_server.shape)))
    print("X_test_server:  {:<20} {:<12}".format(str(X_test_server.shape), str(y_test_server.shape)))
    print("X_data_office:  {:<20} {:<12}".format(str(X_data_office.shape), str(y_data_office.shape)))

    '''
    Generate encodings
    '''
    model = ResNet50AMCA(num_classes,
                         num_features,
                         num_filters=model_filters,
                         activation=activation_fn,
                         ca_decay=ca)
    ckpt  = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              checkpoint_path,
                                              max_to_keep=5)
    ckpt.restore(ckpt_manager.latest_checkpoint)

    X_train_src, X_train_src_acc  = get_acc_encodings(X_train_src, y_train_src)
    X_test_src,  X_test_src_acc   = get_acc_encodings(X_test_src, y_test_src)
    X_train_trg, X_train_trg_acc  = get_acc_encodings(X_train_trg, y_train_trg)
    X_test_trg,  X_test_trg_acc   = get_acc_encodings(X_test_trg, y_test_trg)

    X_train_conf, X_train_conf_acc = get_acc_encodings(X_train_conf, y_train_conf)
    X_test_conf,  X_test_conf_acc  = get_acc_encodings(X_test_conf, y_test_conf)

    X_train_server, X_train_server_acc = get_acc_encodings(X_train_server, y_train_server)
    X_test_server,  X_test_server_acc  = get_acc_encodings(X_test_server, y_test_server)

    X_data_office, X_data_office_acc = get_acc_encodings(X_data_office, y_data_office)

    print("\nEncoding Data shapes:")
    print("X_train_src:    {:<12} {:<10}".format(str(X_train_src.shape), str(y_train_src.shape)))
    print("X_test_src:     {:<12} {:<10}".format(str(X_test_src.shape), str(y_test_src.shape)))
    print("X_train_trg:    {:<12} {:<10}".format(str(X_train_trg.shape), str(y_train_trg.shape)))
    print("X_test_trg:     {:<12} {:<10}".format(str(X_test_trg.shape), str(y_test_trg.shape)))
    print("X_train_conf:   {:<12} {:<10}".format(str(X_train_conf.shape), str(y_train_conf.shape)))
    print("X_test_conf:    {:<12} {:<10}".format(str(X_test_conf.shape), str(y_test_conf.shape)))
    print("X_train_server: {:<12} {:<10}".format(str(X_train_server.shape), str(y_train_server.shape)))
    print("X_test_server:  {:<12} {:<10}".format(str(X_test_server.shape), str(y_test_server.shape)))
    print("X_data_office:  {:<12} {:<10}".format(str(X_data_office.shape), str(y_data_office.shape)))

    kmeans_model, mode_data = generate_kmeans_model(X_train_src, y_train_src,
                                         X_train_server, y_train_server,
                                         num_classes=num_classes)
    print(mode_data)

    X_train_src_acc_kmeans = get_kmeans_acc(kmeans_model, X_train_src, y_train_src)
    X_test_src_acc_kmeans = get_kmeans_acc(kmeans_model, X_test_src, y_test_src)
    X_train_trg_acc_kmeans = get_kmeans_acc(kmeans_model, X_train_trg, y_train_trg)
    X_test_trg_acc_kmeans = get_kmeans_acc(kmeans_model, X_test_trg, y_test_trg)
    X_train_conf_acc_kmeans = get_kmeans_acc(kmeans_model, X_train_conf, y_train_conf)
    X_test_conf_acc_kmeans = get_kmeans_acc(kmeans_model, X_test_conf, y_test_conf)
    X_train_server_acc_kmeans = get_kmeans_acc(kmeans_model, X_train_server, y_train_server, print_labels=True)
    X_test_server_acc_kmeans = get_kmeans_acc(kmeans_model, X_test_server, y_test_server)
    X_data_office_acc_kmeans = get_kmeans_acc(kmeans_model, X_data_office, y_data_office)

    print("\nAccuracies:")
    print("Data            AMCA     Kmeans")
    print("X_train_src:    {:.4f} | {:.4f}".format(X_train_src_acc, X_train_src_acc_kmeans))
    print("X_test_src:     {:.4f} | {:.4f}".format(X_test_src_acc, X_test_src_acc_kmeans))
    print("X_train_trg:    {:.4f} | {:.4f}".format(X_train_trg_acc, X_train_trg_acc_kmeans))
    print("X_test_trg:     {:.4f} | {:.4f}".format(X_test_trg_acc, X_test_trg_acc_kmeans))
    print("X_train_conf:   {:.4f} | {:.4f}".format(X_train_conf_acc, X_train_conf_acc_kmeans))
    print("X_test_conf:    {:.4f} | {:.4f}".format(X_test_conf_acc, X_test_conf_acc_kmeans))
    print("X_train_server: {:.4f} | {:.4f}".format(X_train_server_acc, X_train_server_acc_kmeans))
    print("X_test_server:  {:.4f} | {:.4f}".format(X_test_server_acc, X_test_server_acc_kmeans))
    print("X_data_office:  {:.4f} | {:.4f}".format(X_data_office_acc, X_data_office_acc_kmeans))
