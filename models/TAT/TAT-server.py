repo_path = "/home/kjakkala/mmwave"

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append(os.path.join(repo_path, 'models'))

import h5py
from utils import *
from tqdm import tqdm

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class ConstrictiveRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, x):
        l2_norm = tf.reduce_sum(tf.square(x), axis=0)
        regularization = tf.reduce_mean(l2_norm - tf.reduce_mean(l2_norm)) / 4.0
        return self.scale * regularization


class AMDense(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 **kwargs):

        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]), self.units],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

    def call(self, inputs):
        return tf.matmul(tf.nn.l2_normalize(inputs, -1),
                         tf.nn.l2_normalize(self.kernel, 0))


class Discriminator(tf.keras.Model):

    def __init__(self, num_hidden, num_classes, activation='relu'):
        super().__init__(name='discriminator')
        self.hidden_layers = []
        for dim in num_hidden:
            self.hidden_layers.append(
                tf.keras.layers.Dense(dim, activation=activation))
        self.logits = tf.keras.layers.Dense(num_classes, activation=None)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.logits(x)

        return x


class Classifier(tf.keras.Model):

    def __init__(self,
                 num_hidden,
                 num_classes,
                 activation='relu',
                 ca_decay=1e-3):
        super().__init__(name='classifier')
        self.hidden_layers = []
        for dim in num_hidden[:-1]:
            self.hidden_layers.append(
                tf.keras.layers.Dense(dim, activation=activation))
        self.hidden_layers.append(
            tf.keras.layers.Dense(
                num_hidden[-1],
                activation=activation,
                activity_regularizer=ConstrictiveRegularizer(ca_decay)))
        self.logits = AMDense(
            num_classes,
            kernel_regularizer=ConstrictiveRegularizer(ca_decay),
            name='logits')

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.logits(x)
        return x


def get_cross_entropy_loss(labels, logits):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def AM_logits(labels, logits, m, s):
    cos_theta = tf.clip_by_value(logits, -1, 1)
    phi = cos_theta - m
    adjust_theta = s * tf.where(tf.equal(labels, 1), phi, cos_theta)
    return adjust_theta


def transferable_features(src_features,
                          src_labels,
                          ser_features,
                          K=10,
                          beta=0.001,
                          gamma=0.001,
                          m=0.35,
                          s=10):
    src_features0 = src_features
    ser_features0 = ser_features

    for _ in range(K):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(src_features)
            tape.watch(ser_features)
            tape.watch(src_features0)
            tape.watch(ser_features0)

            src_clas_logits = classifier(src_features, training=False)
            src_clas_logits = AM_logits(labels=src_labels,
                                        logits=src_clas_logits,
                                        m=m,
                                        s=s)
            ser_clas_logits = classifier(ser_features, training=False)

            src_disc_logits = discriminator(src_clas_logits, training=False)
            ser_disc_logits = discriminator(ser_clas_logits, training=False)

            classifier_loss = get_cross_entropy_loss(labels=src_labels,
                                                     logits=src_clas_logits)
            discriminator_loss = get_cross_entropy_loss(
                labels=tf.one_hot(tf.cast(
                    tf.concat([
                        tf.ones(tf.shape(src_disc_logits)[0]),
                        tf.zeros(tf.shape(ser_disc_logits)[0])
                    ],
                              axis=0), tf.uint8),
                                  depth=2),
                logits=tf.concat([src_disc_logits, ser_disc_logits], axis=0))

            l2_src = tf.nn.l2_loss(src_features - src_features0)
            l2_ser = tf.nn.l2_loss(ser_features - ser_features0)

        clas_delta = tape.gradient(classifier_loss, src_features)

        src_disc_delta = tape.gradient(discriminator_loss, src_features)
        ser_disc_delta = tape.gradient(discriminator_loss, ser_features)

        l2_src_delta = tape.gradient(l2_src, src_features)
        l2_ser_delta = tape.gradient(l2_ser, ser_features)

        src_features += (beta * src_disc_delta) - (gamma * l2_src_delta) + (
            beta * clas_delta)
        ser_features += (beta * ser_disc_delta) - (gamma * l2_ser_delta)

    return tf.stop_gradient(src_features), tf.stop_gradient(ser_features)


@tf.function
def test_step(images):
    logits = classifier(images, training=False)
    return tf.nn.softmax(logits)


@tf.function
def train_clas_step(src_enc, src_labels, ser_enc, ser_labels, m, s):
    with tf.GradientTape() as tape:
        #Logits
        src_logits = classifier(src_enc, training=True)
        src_logits_am = AM_logits(labels=src_labels,
                                  logits=src_logits,
                                  m=m,
                                  s=s)
        ser_logits = classifier(ser_enc, training=True)

        src_trans_enc, ser_trans_enc = transferable_features(
            src_enc, src_labels, ser_enc)
        src_trans_logits = classifier(src_trans_enc, training=True)
        src_trans_logits_am = AM_logits(labels=src_labels,
                                        logits=src_trans_logits,
                                        m=m,
                                        s=s)
        ser_trans_logits = classifier(ser_trans_enc, training=True)

        #Loss
        batch_cross_entropy_loss  = get_cross_entropy_loss(labels=src_labels,
                                                           logits=src_trans_logits_am) + \
                                    get_cross_entropy_loss(labels=src_labels,
                                                           logits=src_logits_am)

        batch_l2_loss = tf.reduce_mean(tf.abs(ser_trans_logits - ser_logits))

        total_loss = batch_cross_entropy_loss + batch_l2_loss

    clas_gradients = tape.gradient(total_loss, classifier.trainable_variables)
    clas_optimizer.apply_gradients(
        zip(clas_gradients, classifier.trainable_variables))

    source_train_acc(src_labels, tf.nn.softmax(src_logits))
    server_train_acc(ser_labels, tf.nn.softmax(ser_logits))
    cross_entropy_loss(batch_cross_entropy_loss)
    l2_loss(batch_l2_loss)


@tf.function
def train_disc_step(src_enc, src_labels, ser_enc):
    with tf.GradientTape() as tape:
        src_trans_enc, ser_trans_enc = transferable_features(
            src_enc, src_labels, ser_enc)

        #Logits
        src_logits = classifier(src_enc, training=False)
        ser_logits = classifier(ser_enc, training=False)

        src_trans_logits = classifier(src_trans_enc, training=False)
        ser_trans_logits = classifier(ser_trans_enc, training=False)

        #Disc
        src_disc_logits = discriminator(src_logits, training=True)
        ser_disc_logits = discriminator(ser_logits, training=True)

        src_disc_trans_logits = discriminator(src_trans_logits, training=True)
        ser_disc_trans_logits = discriminator(ser_trans_logits, training=True)

        batch_confusion_loss = get_cross_entropy_loss(labels=tf.one_hot(tf.cast(tf.concat([tf.ones(tf.shape(src_disc_logits)[0]),
                                                                                           tf.zeros(tf.shape(ser_disc_logits)[0])], 0), tf.uint8), 2),
                                                      logits=tf.concat([src_disc_logits,
                                                                        ser_disc_logits], 0)) + \
                               get_cross_entropy_loss(labels=tf.one_hot(tf.cast(tf.concat([tf.ones(tf.shape(src_disc_trans_logits)[0]),
                                                                                           tf.zeros(tf.shape(ser_disc_trans_logits)[0])], 0), tf.uint8), 2),
                                                      logits=tf.concat([src_disc_trans_logits,
                                                                        ser_disc_trans_logits], 0))

    disc_gradients = tape.gradient(batch_confusion_loss,
                                   discriminator.trainable_variables)
    disc_optimizer.apply_gradients(
        zip(disc_gradients, discriminator.trainable_variables))

    confusion_loss(batch_confusion_loss)


dataset_path = os.path.join(repo_path, 'data')
num_classes = 9
batch_size = 64
train_src_days = 3
train_trg_days = 0
train_trg_env_days = 2
epochs = 1000
init_lr = 0.0001
num_features = 256
alpha = 0.05
disc_hidden = [256, 256]
clas_hidden = [256, 256]
activation_fn = 'selu'
m = 0.35
s = 10
notes = "TAT_disc-{}_clas-{}_server-adapt-trans-AMCA".format(
    disc_hidden, clas_hidden)
log_data = "classes-{}_bs-{}_train_src_days-{}_train_trg_days-{}_train_trgenv_days-{}_alpha-{}_initlr-{}_num_feat-{}_act_fn-{}_{}".format(
    num_classes, batch_size, train_src_days, train_trg_days, train_trg_env_days,
    alpha, init_lr, num_features, activation_fn, notes)
log_dir = os.path.join(repo_path, 'logs/TAT/{}'.format(log_data))
encodings_file = os.path.join(repo_path, 'data/encodings.h5')

hf = h5py.File(encodings_file, 'r')

X_train_src = np.array(hf.get('X_train_src'))
y_train_src = np.array(hf.get('y_train_src'))
X_test_src = np.array(hf.get('X_test_src'))
y_test_src = np.array(hf.get('y_test_src'))
X_train_trg = np.array(hf.get('X_train_trg'))
y_train_trg = np.array(hf.get('y_train_trg'))
X_test_trg = np.array(hf.get('X_test_trg'))
y_test_trg = np.array(hf.get('y_test_trg'))

X_train_conf = np.array(hf.get('X_train_conf'))
y_train_conf = np.array(hf.get('y_train_conf'))
X_test_conf = np.array(hf.get('X_test_conf'))
y_test_conf = np.array(hf.get('y_test_conf'))

X_train_server = np.array(hf.get('X_train_server'))
y_train_server = np.array(hf.get('y_train_server'))
X_test_server = np.array(hf.get('X_test_server'))
y_test_server = np.array(hf.get('y_test_server'))

X_data_office = np.array(hf.get('X_data_office'))
y_data_office = np.array(hf.get('y_data_office'))

hf.close()

print("Final shapes: ")
print(X_train_src.shape, y_train_src.shape, X_test_src.shape, y_test_src.shape,
      X_train_trg.shape, y_train_trg.shape, X_test_trg.shape, y_test_trg.shape)
print(X_train_conf.shape, y_train_conf.shape, X_test_conf.shape,
      y_test_conf.shape)
print(X_train_server.shape, y_train_server.shape, X_test_server.shape,
      y_test_server.shape)
print(X_data_office.shape, y_data_office.shape)

#Test
conf_test_set = tf.data.Dataset.from_tensor_slices((X_test_conf, y_test_conf))
conf_test_set = conf_test_set.batch(batch_size, drop_remainder=False)
conf_test_set = conf_test_set.prefetch(batch_size)

server_test_set = tf.data.Dataset.from_tensor_slices(
    (X_test_server, y_test_server))
server_test_set = server_test_set.batch(batch_size, drop_remainder=False)
server_test_set = server_test_set.prefetch(batch_size)

office_test_set = tf.data.Dataset.from_tensor_slices(
    (X_data_office, y_data_office))
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

server_train_set = tf.data.Dataset.from_tensor_slices(
    (X_train_server, y_train_server))
server_train_set = server_train_set.shuffle(X_train_server.shape[0])
server_train_set = server_train_set.batch(batch_size, drop_remainder=True)
server_train_set = server_train_set.prefetch(batch_size)
server_train_set = server_train_set.repeat(-1)

l2_loss = tf.keras.metrics.Mean(name='l2_loss')
confusion_loss = tf.keras.metrics.Mean(name='confusion_loss')
cross_entropy_loss = tf.keras.metrics.Mean(name='cross_entropy_loss')
temporal_test_acc = tf.keras.metrics.CategoricalAccuracy(
    name='temporal_test_acc')
source_train_acc = tf.keras.metrics.CategoricalAccuracy(name='source_train_acc')
source_test_acc = tf.keras.metrics.CategoricalAccuracy(name='source_test_acc')
office_test_acc = tf.keras.metrics.CategoricalAccuracy(name='office_test_acc')
server_test_acc = tf.keras.metrics.CategoricalAccuracy(name='server_test_acc')
server_train_acc = tf.keras.metrics.CategoricalAccuracy(name='server_train_acc')
conference_test_acc = tf.keras.metrics.CategoricalAccuracy(
    name='conference_test_acc')

learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
    init_lr, decay_steps=5000, end_learning_rate=init_lr * 1e-2, cycle=True)
discriminator = Discriminator(disc_hidden, 2, activation_fn)
classifier = Classifier(clas_hidden, num_classes, activation_fn)

disc_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                          beta_1=0.5)
clas_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                          beta_1=0.5)

summary_writer = tf.summary.create_file_writer(log_dir)

# In[ ]:

m_anneal = tf.Variable(0, dtype="float32")
for epoch in tqdm(range(epochs)):
    m_anneal.assign(tf.minimum(m * ((epoch * 2) / 1000.0), m))
    for source_data, server_data in zip(src_train_set, server_train_set):
        train_clas_step(source_data[0], source_data[1], server_data[0],
                        server_data[1], s, m_anneal)
        train_disc_step(source_data[0], source_data[1], server_data[0])

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
        tf.summary.scalar("cross_entropy_loss",
                          cross_entropy_loss.result(),
                          step=epoch)
        tf.summary.scalar("temporal_test_acc",
                          temporal_test_acc.result(),
                          step=epoch)
        tf.summary.scalar("source_train_acc",
                          source_train_acc.result(),
                          step=epoch)
        tf.summary.scalar("source_test_acc",
                          source_test_acc.result(),
                          step=epoch)
        tf.summary.scalar("office_test_acc",
                          office_test_acc.result(),
                          step=epoch)
        tf.summary.scalar("server_test_acc",
                          server_test_acc.result(),
                          step=epoch)
        tf.summary.scalar("server_train_acc",
                          server_train_acc.result(),
                          step=epoch)
        tf.summary.scalar("conference_test_acc",
                          conference_test_acc.result(),
                          step=epoch)
        tf.summary.scalar("confusion_loss", confusion_loss.result(), step=epoch)
        tf.summary.scalar("l2_loss", l2_loss.result(), step=epoch)

    cross_entropy_loss.reset_states()
    temporal_test_acc.reset_states()
    source_train_acc.reset_states()
    source_test_acc.reset_states()
    office_test_acc.reset_states()
    server_test_acc.reset_states()
    server_train_acc.reset_states()
    conference_test_acc.reset_states()
    confusion_loss.reset_states()
    l2_loss.reset_states()
