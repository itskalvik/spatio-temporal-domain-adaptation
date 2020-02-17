import os
repo_path = os.getenv('MMWAVE_PATH')
import sys
sys.path.append(os.path.join(repo_path, 'models'))
from utils import *
from resnet_amca import ResNet50AMCA
import numpy as np
import argparse
import h5py

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

def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--num_features', type=int, default=128)
    parser.add_argument('--model_filters', type=int, default=64)
    parser.add_argument('--activation_fn', default='selu')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--train_source_days', type=int, default=3)
    parser.add_argument('--train_source_unlabeled_days', type=int, default=0)
    parser.add_argument('--train_server_days', type=int, default=0)
    parser.add_argument('--train_conference_days', type=int, default=0)
    parser.add_argument('--save_freq', type=int, default=25)
    parser.add_argument('--checkpoint_path', default="checkpoints")
    parser.add_argument('--summary_writer_path', default="tensorboard_logs")
    parser.add_argument('--anneal', type=int, default=4)
    parser.add_argument('--s', type=int, default=10)
    parser.add_argument('--m', type=float, default=0.1)
    parser.add_argument('--ca', type=float, default=1e-3)
    parser.add_argument('--log_dir', default="logs/Baselines/AMCA/")
    parser.add_argument('--notes', default="AMCABaseline")
    parser.add_argument('--enc_path', default="data/encodings-server.h5")
    return parser

if __name__=='__main__':
    parser = get_parser()
    arg = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES']=arg.gpu

    dataset_path    = os.path.join(repo_path, 'data')
    num_classes     = arg.num_classes
    batch_size      = arg.batch_size
    train_source_days = arg.train_source_days
    train_server_days = arg.train_server_days
    train_conference_days = arg.train_conference_days
    train_source_unlabeled_days = arg.train_source_unlabeled_days
    save_freq       = arg.save_freq
    epochs          = arg.epochs
    init_lr         = arg.init_lr
    num_features    = arg.num_features
    activation_fn   = arg.activation_fn
    model_filters   = arg.model_filters
    anneal          = arg.anneal
    s               = arg.s
    m               = arg.m
    ca              = arg.ca

    run_params      = dict(vars(arg))
    del run_params['log_dir']
    del run_params['checkpoint_path']
    del run_params['summary_writer_path']
    del run_params['gpu']
    del run_params['save_freq']
    del run_params['enc_path']
    sorted(run_params)

    run_params      = str(run_params).replace(" ", "").replace("'", "").replace(",", "-")[1:-1]
    log_dir         = os.path.join(repo_path, arg.log_dir, run_params)
    arg.log_dir     = log_dir

    summary_writer_path = os.path.join(log_dir, arg.summary_writer_path)
    checkpoint_path = os.path.join(log_dir, arg.checkpoint_path)
    encodings_file  = os.path.join(repo_path, arg.enc_path)

    '''
    Data Preprocessing
    '''

    X_data, y_data, classes = get_h5dataset(os.path.join(dataset_path, 'source_data.h5'))
    print(X_data.shape, y_data.shape, "\n", classes)

    X_data, y_data = balance_dataset(X_data, y_data,
                                     num_days=10,
                                     num_classes=len(classes),
                                     max_samples_per_class=95)
    print(X_data.shape, y_data.shape)

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
    print("Final shapes: ")
    print(X_train_src.shape, y_train_src.shape,  X_test_src.shape, \
          y_test_src.shape, X_train_trg.shape, y_train_trg.shape, \
          X_test_trg.shape, y_test_trg.shape)

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

    print(X_train_conf.shape, y_train_conf.shape,
          X_test_conf.shape, y_test_conf.shape, "\n",
          X_train_server.shape, y_train_server.shape,
          X_test_server.shape, y_test_server.shape, "\n",
          X_data_office.shape,  y_data_office.shape)

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

    print("Final accuracies and shapes: ")
    print("X_train_src:    {:.4f} | {:<12} {:<10}".format(X_train_src_acc, str(X_train_src.shape),    str(y_train_src.shape)))
    print("X_test_src:     {:.4f} | {:<12} {:<10}".format(X_test_src_acc, str(X_test_src.shape),     str(y_test_src.shape)))
    print("X_train_trg:    {:.4f} | {:<12} {:<10}".format(X_train_trg_acc, str(X_train_trg.shape),    str(y_train_trg.shape)))
    print("X_test_trg:     {:.4f} | {:<12} {:<10}".format(X_test_trg_acc, str(X_test_trg.shape),     str(y_test_trg.shape)))
    print("X_train_conf:   {:.4f} | {:<12} {:<10}".format(X_train_conf_acc, str(X_train_conf.shape),   str(y_train_conf.shape)))
    print("X_test_conf:    {:.4f} | {:<12} {:<10}".format(X_test_conf_acc, str(X_test_conf.shape),    str(y_test_conf.shape)))
    print("X_train_server: {:.4f} | {:<12} {:<10}".format(X_train_server_acc, str(X_train_server.shape), str(y_train_server.shape)))
    print("X_test_server:  {:.4f} | {:<12} {:<10}".format(X_test_server_acc, str(X_test_server.shape),  str(y_test_server.shape)))
    print("X_data_office:  {:.4f} | {:<12} {:<10}".format(X_data_office_acc, str(X_data_office.shape),  str(y_data_office.shape)))

    hf = h5py.File(encodings_file, 'w')

    hf.create_dataset('X_train_src', data=X_train_src)
    hf.create_dataset('y_train_src', data=y_train_src)
    hf.create_dataset('X_test_src', data=X_test_src)
    hf.create_dataset('y_test_src', data=y_test_src)
    hf.create_dataset('X_train_trg', data=X_train_trg)
    hf.create_dataset('y_train_trg', data=y_train_trg)
    hf.create_dataset('X_test_trg', data=X_test_trg)
    hf.create_dataset('y_test_trg', data=y_test_trg)

    hf.create_dataset('X_train_conf', data=X_train_conf)
    hf.create_dataset('y_train_conf', data=y_train_conf)
    hf.create_dataset('X_test_conf', data=X_test_conf)
    hf.create_dataset('y_test_conf', data=y_test_conf)

    hf.create_dataset('X_train_server', data=X_train_server)
    hf.create_dataset('y_train_server', data=y_train_server)
    hf.create_dataset('X_test_server', data=X_test_server)
    hf.create_dataset('y_test_server', data=y_test_server)

    hf.create_dataset('X_data_office', data=X_data_office)
    hf.create_dataset('y_data_office', data=y_data_office)

    hf.close()
