import random
from config import *
from tensorflow.contrib.framework import arg_scope
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, xavier_initializer

initializer = xavier_initializer()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_layer(input, filter, kernel, stride=1, layer_name="conv", padding='SAME'):
    with tf.name_scope(layer_name):
        conv = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding, kernel_initializer=initializer)
        return  conv

def Linear(x, out_length, layer_name) :
    with tf.name_scope(layer_name):
        linear = tf.layers.dense(inputs=x, units=out_length, kernel_initializer=initializer)
        return linear

def Batch_Normalization(x, training, scope="batch"):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def SoftMax(x, axis=-1):
    return tf.nn.softmax(x, axis=axis)

def Max_Pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Global_Average_Pooling(x):
     return global_avg_pool(x, name='Global_avg_pooling')

def flip_gradient(x, l=1.0):
	positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
	negative_path = -x * tf.cast(l, tf.float32)
	return positive_path + negative_path

def Soft_Assignment(z_, x_, n_book, alpha):

    x = tf.split(x_, n_book, 1)
    y = tf.split(z_, n_book, 1)
    for i in range(n_book):
        size_x = tf.shape(x[i])[0]
        size_y = tf.shape(y[i])[0]
        xx = tf.expand_dims(x[i], -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y]))

        yy = tf.expand_dims(y[i], -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])

        diff = 1-tf.reduce_sum(tf.multiply(xx,yy), 1)
        softmax_diff = SoftMax(diff * (-alpha), 1)

        if i==0:
            soft_des_tmp = tf.matmul(softmax_diff, y[i], transpose_a=False, transpose_b=False)
            descriptor = soft_des_tmp
        else:
            soft_des_tmp = tf.matmul(softmax_diff, y[i], transpose_a=False, transpose_b=False)
            descriptor = tf.concat([descriptor, soft_des_tmp], axis=1)

    return descriptor

def Intra_Norm(features, numSeg):
    x = tf.split(features, numSeg, 1)
    for i in range(numSeg):
        norm_tmp = tf.nn.l2_normalize(x[i], axis=1)
        if i==0:
            innorm = norm_tmp
        else:
            innorm = tf.concat([innorm, norm_tmp], axis=1)
    return innorm

# N_pair Product Quantization loss
def N_PQ_loss(labels_Similarity, embeddings_x, embeddings_q, reg_lambda=0.002):

  reg_anchor = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_x), 1))
  reg_positive = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings_q), 1))
  l2loss = tf.multiply(0.25 * reg_lambda, reg_anchor + reg_positive, name='l2loss')

  FQ_Similarity = tf.matmul(embeddings_x, embeddings_q, transpose_a=False, transpose_b=True)

  # Add the softmax loss.
  loss = tf.nn.softmax_cross_entropy_with_logits(logits=FQ_Similarity, labels=labels_Similarity)
  loss = tf.reduce_mean(loss, name='xentropy')

  return l2loss + loss

def CLS_loss(label, logits):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
    return loss

# Subspace Minimax Entropy loss
def SME_loss(features, Centroids, numSeg):

    x = tf.split(features, numSeg, 1)
    y = tf.split(Centroids, numSeg, 0)

    for i in range(numSeg):
        tmp = tf.expand_dims(tf.matmul(x[i], y[i]), axis=-1)
        if i==0:
            logits = tmp
        else:
            logits = tf.concat([logits, tmp], axis=2)

    logits = SoftMax(tf.reduce_mean(logits, axis=2), axis=1)
    loss = tf.reduce_mean(tf.reduce_sum(logits*(tf.log(logits + 1e-5)), 1))
    return loss

def data_augmentation(batch, crop_size=32):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [crop_size, crop_size], 4)
    return batch

def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch

def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch