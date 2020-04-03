from utils.Functions import *

with tf.variable_scope("placeholder"):
    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x')
    x_T = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x_T')
    label = tf.placeholder(tf.float32, shape=[None, n_CLASSES], name='label')
    label_Mat = tf.placeholder(tf.float32, shape=[None, None], name='label_Mat')
    training_flag = tf.placeholder(tf.bool, name='training_flag')
    global_step = tf.placeholder(tf.float32, name='global_step')

class GPQ():
    def __init__(self, training):
        self.training = training
        self.Z = tf.get_variable('Z', [intn_word, len_code * n_book], dtype=tf.float32,
                                   initializer=initializer, trainable=True)
        self.Prototypes = tf.get_variable('C', [n_CLASSES, len_code * n_book], dtype=tf.float32,
                                         initializer=initializer, trainable=True)

    #Feature Extractor
    def F(self, input_x):
        with tf.variable_scope('Fixed_VGG', reuse=tf.AUTO_REUSE):
            x = conv_layer(input_x, filter=64, kernel=[3, 3], stride=1, layer_name='conv0')
            x = Batch_Normalization(x, training=self.training, scope='batch0')
            x = Relu(x)
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name='conv0-1')
            x = Batch_Normalization(x, training=self.training, scope='batch0-1')
            x = Relu(x)
            x = Max_Pooling(x, pool_size=[2, 2], stride=2)

            x = conv_layer(x, filter=128, kernel=[3, 3], stride=1, layer_name='conv1')
            x = Batch_Normalization(x, training=self.training, scope='batch1')
            x = Relu(x)
            x = conv_layer(x, filter=128, kernel=[3, 3], stride=1, layer_name='conv1-1')
            x = Batch_Normalization(x, training=self.training, scope='batch1-1')
            x = Relu(x)
            x = Max_Pooling(x, pool_size=[2, 2], stride=2)

            x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2')
            x = Batch_Normalization(x, training=self.training, scope='batch2')
            x = Relu(x)
            x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2-1')
            x = Batch_Normalization(x, training=self.training, scope='batch2-1')
            x = Relu(x)
            x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2-2')
            x = Batch_Normalization(x, training=self.training, scope='batch2-2')
            x = Relu(x)
            x = Max_Pooling(x, pool_size=[2, 2], stride=2)
            x_branch = Global_Average_Pooling(x)

            x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3')
            x = Batch_Normalization(x, training=self.training, scope='batch3')
            x = Relu(x)
            x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3-1')
            x = Batch_Normalization(x, training=self.training, scope='batch3-1')
            x = Relu(x)
            x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3-2')
            x = Batch_Normalization(x, training=self.training, scope='batch3-2')
            x = Relu(x)

            x = Global_Average_Pooling(x)
            x = tf.concat([x, x_branch], 1)

            x = Linear(x, len_code * n_book, layer_name='feature_vector')

        return x

    #Classifier
    def C(self, features, Prototypes):
        with tf.variable_scope('C', reuse=tf.AUTO_REUSE):
            x = tf.split(features, n_book, 1)
            y = tf.split(Prototypes, n_book, 0)
            for i in range(n_book):
                sub_res = tf.expand_dims(tf.matmul(x[i], y[i]), axis=-1)
                if i == 0:
                    res = sub_res
                else:
                    res = tf.concat([res, sub_res], axis=2)
            logits = tf.reduce_mean(res, axis=2)
        return logits