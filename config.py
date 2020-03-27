import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import pickle
import scipy.io
from scipy.sparse import csr_matrix


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

data_dir = './cifar10'
ImagNet_pretrained_path = './ImageNet_pretrained'
cifar10_label_sim_path = './cifar10/cifar10_Similarity.mat'
model_save_path = './models/'
# load trained model
model_load_path = './models/48bits_example.ckpt'

'Dataset info'
# Source: 5,000, Target: 54,000
# Gallery: 54,000 Query: 1,000

n_CLASSES = 10
image_size = 32
img_channels = 3
n_DB = 54000

'Hyperparameters for training'
# Training epochs, 1 epoch represents training all the source data once
total_epochs = 300
batch_size = 500
# save model for every save_term-th epoch
save_term = 20

# length of codeword
len_code = 12

# Number of codebooks
n_book = 12

# Number of codewords=(2^bn_word)
bn_word = 4
intn_word = pow(2, bn_word)

# Number of bits for retrieval
n_bits = n_book * bn_word

# Soft assignment input scaling factor
alpha = 20.0

# Classification input scaling factor
beta = 4

# lam1, 2: loss function balancing parameters
lam_1 = 0.1
lam_2 = 0.1


