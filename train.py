from utils.GPQ_network import *
from utils.Functions import *
from utils import cifar10 as ci10
from utils.RetrievalTest import *

def run():
    print("num_Codewords: 2^%d, num_Codebooks: %d, Bits: %d" % (bn_word, n_book, n_bits))
    Source_x, Source_y, Target_x = ci10.prepare_data(data_dir, True)

    Gallery_x, Query_x = ci10.prepare_data(data_dir, False)

    Net = GPQ(training=training_flag)
    Prototypes = Intra_Norm(Net.Prototypes, n_book)
    Z = Soft_Assignment(Prototypes, Net.Z, n_book, alpha)

    feature_S = Net.F(x)
    feature_T = flip_gradient(Net.F(x_T))

    feature_S = Intra_Norm(feature_S, n_book)
    feature_T = Intra_Norm(feature_T, n_book)

    descriptor_S = Soft_Assignment(Z, feature_S, n_book, alpha)

    logits_S = Net.C(feature_S * beta, tf.transpose(Prototypes) * beta)

    hash_loss = N_PQ_loss(labels_Similarity=label_Mat, embeddings_x=feature_S, embeddings_q=descriptor_S)
    cls_loss = CLS_loss(label, logits_S)
    entropy_loss = SME_loss(feature_T * beta, tf.transpose(Prototypes) * beta, n_book)

    cost = hash_loss + lam_1*entropy_loss + lam_2*cls_loss

    pretrained_mat = scipy.io.loadmat(ImagNet_pretrained_path)

    var_F = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Fixed_VGG')

    decayed_lr = tf.train.exponential_decay(0.0002, global_step, 100, 0.95, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=decayed_lr, beta1=0.5).minimize(loss=cost)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        print("Load ImageNet2012 pretrained model")
        for i in range(len(var_F) - 2):
            sess.run(var_F[i].assign(np.squeeze(pretrained_mat[var_F[i].name])))

        total_iter = 0

        for epoch in range(1, total_epochs + 1):

            if epoch == 1:

                label_Similarity = csr_matrix(scipy.io.loadmat(cifar10_label_sim_path)['label_Similarity'])
                label_Similarity = label_Similarity.todense()

                num_S = np.shape(Source_x)[0]
                num_T = np.shape(Target_x)[0]

                iteration = int(num_S / batch_size)

            for step in range(iteration):

                total_iter += 1

                rnd_idx_S = np.random.choice(num_S, size=batch_size, replace=False)

                batch_Sx = Source_x[rnd_idx_S]
                batch_Sy = Source_y[rnd_idx_S]

                batch_Sy = np.eye(n_CLASSES)[batch_Sy]
                batch_Sy_Mat = np.matmul(batch_Sy, batch_Sy.transpose())
                batch_Sy_Mat /= np.sum(batch_Sy_Mat, axis=1, keepdims=True)

                batch_Sx = data_augmentation(batch_Sx)

                rnd_idx_T = np.random.choice(num_T, size=batch_size, replace=False)
                batch_Tx = Target_x[rnd_idx_T]
                batch_Tx = data_augmentation(batch_Tx)

                _, batch_loss, batch_entropy_loss, batch_closs, batch_hash_loss, batch_lr = sess.run(
                    [train_op, cost, entropy_loss, cls_loss, hash_loss, decayed_lr],
                    feed_dict={x: batch_Sx, label: batch_Sy, label_Mat: batch_Sy_Mat, x_T: batch_Tx,
                               training_flag: True, global_step: total_iter-1})

                if (total_iter) % 10 == 0:
                    print("epoch: %d/%d, iter: %d - (Batch) loss: %.4f, hash: %.4f, cls: %.4f, ent: %.4f, lr: %.5f" % (
                            epoch, total_epochs, total_iter, batch_loss, batch_hash_loss, batch_closs,
                            batch_entropy_loss, batch_lr))
            if epoch % save_term == 0:
                print('Model saved at %d'%(epoch))
                saver.save(sess=sess, save_path=model_save_path+'%d.ckpt'%(epoch))

if __name__ == '__main__':
    run()
