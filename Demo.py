from utils.GPQ_network import *
from utils.Functions import *
from utils import cifar10 as ci10
from utils.RetrievalTest import *

def run():
    print("num_Codewords: 2^%d, num_Codebooks: %d, Bits: %d" % (bn_word, n_book, n_bits))
    Gallery_x, Query_x = ci10.prepare_data(data_dir, False)

    label_Similarity = csr_matrix(scipy.io.loadmat(cifar10_label_sim_path)['label_Similarity'])
    label_Similarity = label_Similarity.todense()

    Net = GPQ(training=training_flag)
    feature = Net.F(x)
    Z = Net.Z
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=config) as sess:
        saver.restore(sess, model_load_path)
        mAP = PQ_retrieval(sess, x, training_flag, feature, Z, n_book, Gallery_x, Query_x, label_Similarity, True, TOP_K=n_DB)
        print(model_load_path+" mAP: %.4f"%(mAP))

if __name__ == '__main__':
    run()