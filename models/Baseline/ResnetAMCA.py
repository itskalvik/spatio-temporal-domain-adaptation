repo_path = "/users/kjakkala/mmwave"

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import argparse
import sys
sys.path.append(os.path.join(repo_path, 'models'))

from utils import *
from resnet_amca import ResNet50AMCA
from pix2pix import upsample

import tensorflow as tf
print(tf.__version__)

def get_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--s', type=int, default=10)
    parser.add_argument('--m', type=float, default=0.1)
    parser.add_argument('--ca', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--num_features', type=int, default=256)
    parser.add_argument('--activation_fn', default='selu')
    return parser

parser = get_parser()
arg = parser.parse_args()

dataset_path    = os.path.join(repo_path, 'data')
num_classes     = 9
batch_size      = 64
train_src_days  = 3
train_trg_days  = 0
train_trg_env_days = 0
epochs          = arg.epochs
init_lr         = arg.init_lr
num_features    = arg.num_features
activation_fn   = arg.activation_fn
s               = arg.s
m               = arg.m
ca              = arg.ca
notes           = "AM_S-{}_M-{}_CA-{}_Baseline".format(s, m, ca)
log_data = "classes-{}_bs-{}_train_src_days-{}_train_trg_days-{}_train_trgenv_days-{}_initlr-{}_num_feat-{}_act_fn-{}_{}".format(num_classes,
                                                                                                                                 batch_size,
                                                                                                                                 train_src_days,
                                                                                                                                 train_trg_days,
                                                                                                                                 train_trg_env_days,
                                                                                                                                 init_lr,
                                                                                                                                 num_features,
                                                                                                                                 activation_fn,
                                                                                                                                 notes)
log_dir         = os.path.join(repo_path, 'logs/Baselines/AMCA/{}'.format(log_data))

X_data, y_data, classes = get_h5dataset(os.path.join(dataset_path, 'source_data.h5'))
X_data = resize_data(X_data)
print(X_data.shape, y_data.shape, "\n", classes)

X_data, y_data = balance_dataset(X_data, y_data,
                                 num_days=10,
                                 num_classes=len(classes),
                                 max_samples_per_class=95)
print(X_data.shape, y_data.shape)

#remove harika's data (incomplete data)
X_data = np.delete(X_data, np.where(y_data[:, 0] == 1)[0], 0)
y_data = np.delete(y_data, np.where(y_data[:, 0] == 1)[0], 0)

#update labes to handle 9 classes instead of 10
y_data[y_data[:, 0] >= 2, 0] -= 1
del classes[1]
print(X_data.shape, y_data.shape, "\n", classes)

#split days of data to train and test
X_src = X_data[y_data[:, 1] < train_src_days]
y_src = y_data[y_data[:, 1] < train_src_days, 0]
y_src = np.eye(len(classes))[y_src]
X_train_src, X_test_src, y_train_src, y_test_src = train_test_split(X_src,
                                                                    y_src,
                                                                    stratify=y_src,
                                                                    test_size=0.10,
                                                                    random_state=42)

X_trg = X_data[y_data[:, 1] >= train_src_days]
y_trg = y_data[y_data[:, 1] >= train_src_days]
X_train_trg = X_trg[y_trg[:, 1] < train_src_days+train_trg_days]
y_train_trg = y_trg[y_trg[:, 1] < train_src_days+train_trg_days, 0]
y_train_trg = np.eye(len(classes))[y_train_trg]

X_test_trg = X_data[y_data[:, 1] >= train_src_days+train_trg_days]
y_test_trg = y_data[y_data[:, 1] >= train_src_days+train_trg_days, 0]
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
print(X_train_src.shape, y_train_src.shape,  X_test_src.shape, y_test_src.shape, X_train_trg.shape, y_train_trg.shape, X_test_trg.shape, y_test_trg.shape)

X_train_conf,   y_train_conf,   X_test_conf,   y_test_conf   = get_trg_data(os.path.join(dataset_path, 'target_conf_data.h5'),   classes, train_trg_env_days)
X_train_server, y_train_server, X_test_server, y_test_server = get_trg_data(os.path.join(dataset_path, 'target_server_data.h5'), classes, train_trg_env_days)
_             , _             , X_data_office, y_data_office = get_trg_data(os.path.join(dataset_path, 'target_office_data.h5'), classes, 0)

print(X_train_conf.shape,   y_train_conf.shape,    X_test_conf.shape,   y_test_conf.shape)
print(X_train_server.shape, y_train_server.shape,  X_test_server.shape, y_test_server.shape)
print(X_data_office.shape,  y_data_office.shape)

#get tf.data objects for each set

#Test
conf_test_set = tf.data.Dataset.from_tensor_slices((X_test_conf, y_test_conf))
conf_test_set = conf_test_set.batch(batch_size, drop_remainder=False)
conf_test_set = conf_test_set.prefetch(batch_size)

server_test_set = tf.data.Dataset.from_tensor_slices((X_test_server, y_test_server))
server_test_set = server_test_set.batch(batch_size, drop_remainder=False)
server_test_set = server_test_set.prefetch(batch_size)

office_test_set = tf.data.Dataset.from_tensor_slices((X_data_office, y_data_office))
office_test_set = office_test_set.batch(batch_size, drop_remainder=False)
office_test_set = office_test_set.prefetch(batch_size)

src_test_set = tf.data.Dataset.from_tensor_slices((X_test_src, y_test_src))
src_test_set = src_test_set.batch(batch_size, drop_remainder=False)
src_test_set = src_test_set.prefetch(batch_size)

time_test_set = tf.data.Dataset.from_tensor_slices((X_test_trg, y_test_trg))
time_test_set = time_test_set.batch(batch_size, drop_remainder=False)
time_test_set = time_test_set.prefetch(batch_size)

#Train
src_train_set = tf.data.Dataset.from_tensor_slices((X_train_src, y_train_src))
src_train_set = src_train_set.shuffle(X_train_src.shape[0])
src_train_set = src_train_set.batch(batch_size, drop_remainder=True)
src_train_set = src_train_set.prefetch(batch_size)

def get_cross_entropy_loss(labels, logits):
  loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
  return tf.reduce_mean(loss)

def AM_logits(labels, logits, m, s):
  cos_theta = tf.clip_by_value(logits, -1,1)
  phi = cos_theta - m
  adjust_theta = s * tf.where(tf.equal(labels,1), phi, cos_theta)
  return adjust_theta

source_train_acc     = tf.keras.metrics.CategoricalAccuracy(name='source_train_acc')
source_test_acc      = tf.keras.metrics.CategoricalAccuracy(name='source_test_acc')
office_test_acc      = tf.keras.metrics.CategoricalAccuracy(name='office_test_acc')
server_test_acc      = tf.keras.metrics.CategoricalAccuracy(name='server_test_acc')
temporal_test_acc    = tf.keras.metrics.CategoricalAccuracy(name='temporal_test_acc')
conference_test_acc  = tf.keras.metrics.CategoricalAccuracy(name='conference_test_acc')
cross_entropy_loss   = tf.keras.metrics.Mean(name='cross_entropy_loss')

@tf.function
def test_step(images):
  logits = model(images, training=False)
  return tf.nn.softmax(logits)

@tf.function
def train_step(src_images, src_labels, s, m):
  with tf.GradientTape() as tape:
    src_logits = model(src_images, training=True)
    src_logits = AM_logits(labels=src_labels, logits=src_logits, m=m, s=s)
    batch_cross_entropy_loss  = get_cross_entropy_loss(labels=src_labels,
                                                       logits=src_logits)
    total_loss = batch_cross_entropy_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  source_train_acc(src_labels, tf.nn.softmax(src_logits))
  cross_entropy_loss(batch_cross_entropy_loss)

learning_rate  = tf.keras.optimizers.schedules.PolynomialDecay(init_lr,
                                                               decay_steps=(X_train_src.shape[0]//batch_size)*200,
                                                               end_learning_rate=init_lr*1e-2,
                                                               cycle=True)
model      = ResNet50AMCA(num_classes, num_features, activation=activation_fn, ca_decay=ca)
optimizer  = tf.keras.optimizers.Adam(learning_rate = learning_rate)

summary_writer = tf.summary.create_file_writer(log_dir)

m_anneal = tf.Variable(0, dtype="float32")
for epoch in range(epochs):
  m_anneal.assign(tf.minimum(m*((epoch*2)/1000.0), m))

  for source_data in src_train_set:
    train_step(source_data[0], source_data[1], s, m_anneal)

  for data in time_test_set:
    temporal_test_acc(test_step(data[0]), data[1])

  for data in src_test_set:
    source_test_acc(test_step(data[0]), data[1])

  for data in office_test_set:
    office_test_acc(test_step(data[0]), data[1])

  for data in server_test_set:
    server_test_acc(test_step(data[0]), data[1])

  for data in conf_test_set:
    conference_test_acc(test_step(data[0]), data[1])

  with summary_writer.as_default():
    tf.summary.scalar("temporal_test_acc", temporal_test_acc.result(), step=epoch)
    tf.summary.scalar("source_train_acc", source_train_acc.result(), step=epoch)
    tf.summary.scalar("source_test_acc", source_test_acc.result(), step=epoch)
    tf.summary.scalar("office_test_acc", office_test_acc.result(), step=epoch)
    tf.summary.scalar("server_test_acc", server_test_acc.result(), step=epoch)
    tf.summary.scalar("conference_test_acc", conference_test_acc.result(), step=epoch)
    tf.summary.scalar("cross_entropy_loss", cross_entropy_loss.result(), step=epoch)

  temporal_test_acc.reset_states()
  source_train_acc.reset_states()
  source_test_acc.reset_states()
  office_test_acc.reset_states()
  server_test_acc.reset_states()
  conference_test_acc.reset_states()
  cross_entropy_loss.reset_states()
