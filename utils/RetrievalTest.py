from config import *

# Average Precision (AP) Calculation
def cat_apcal(label_Similarity, IX, top_N):

    [_, numtest] = IX.shape

    apall = np.zeros(numtest)

    for i in range(numtest):
        y = IX[:, i]
        x = 0
        p = 0

        for j in range(top_N):
            if label_Similarity[i, y[j]] == 1:
                x = x + 1
                p = p + float(x) / (j + 1)
        if p == 0:
            apall[i] = 0
        else:
            apall[i] = p / x

    mAP = np.mean(apall)

    return mAP

# Find the closest codeword index
def Indexing(Z, descriptor, numSeg):

    x = tf.split(descriptor, numSeg, 1)
    y = tf.split(Z, numSeg, 1)
    for i in range(numSeg):
        size_x = tf.shape(x[i])[0]
        size_y = tf.shape(y[i])[0]
        xx = tf.expand_dims(x[i], -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y]))

        yy = tf.expand_dims(y[i], -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])
        diff = tf.reduce_sum(tf.multiply(xx,yy), 1)

        arg = tf.argmax(diff, 1)
        max_idx = tf.reshape(arg, [-1, 1])

        if i == 0:
            quant_idx = max_idx
        else:
            quant_idx = tf.concat([quant_idx, max_idx], axis=1)
    return quant_idx

# Compute distances and build look-up-table
def pqDist(Z, numSeg, g_x, q_x):
    n1 = q_x.shape[0]
    n2 = g_x.shape[0]
    l1, l2 = Z.shape

    D_Z = np.zeros((l1, numSeg), dtype=np.float32)

    q_x_split = np.split(q_x, numSeg, 1)
    g_x_split = np.split(g_x, numSeg, 1)
    Z_split = np.split(Z, numSeg, 1)
    D_Z_split = np.split(D_Z, numSeg, 1)

    Dpq = np.zeros((n1, n2), dtype=np.float32)

    for i in range(n1):
        for j in range(numSeg):
            for k in range(l1):
                D_Z_split[j][k] =1-np.dot(q_x_split[j][i],Z_split[j][k])
            if j == 0:
                y = D_Z_split[j][g_x_split[j]]
            else:
                y = np.add(y, D_Z_split[j][g_x_split[j]])
        Dpq[i, :] = np.squeeze(y)
    return Dpq

def color_deprocessing_toshow(x_):

    #Return to original data
    x_ = np.squeeze(x_)

    x_[:, :, 0] = (x_[:, :, 0]* 63.01 + 125.642)/255
    x_[:, :, 1] = (x_[:, :, 1]* 62.157 + 123.738)/255
    x_[:, :, 2] = (x_[:, :, 2]* 66.94 + 114.46)/255

    return x_

# Do image retrieval using PQ table
def PQ_retrieval(sess, x, training_flag, feature, Z, n_book, db_x, test_x, label_Similarity, visual_flag, TOP_K=54000):
    print("Do retrieval")
    pre_index = 0
    test_pre_index = 0
    iteration = 100
    test_iteration = 5
    train_data_num = np.shape(db_x)[0]
    test_data_num = np.shape(test_x)[0]
    batch_size = int(train_data_num / iteration)
    batch_size_test = int(test_data_num / test_iteration)

    idxed_descriptor = Indexing(Z, feature, n_book)

    for step in range(iteration + 1):
        if pre_index + batch_size < train_data_num:
            batch_x = db_x[pre_index: pre_index + batch_size]
        else:
            batch_x = db_x[pre_index:]
        pre_index += batch_size
        if np.shape(batch_x)[0] == 0:
            continue
        retrieval_feed_dict_train = {
            x: batch_x,
            training_flag: False
        }
        train_features_batch = sess.run(idxed_descriptor, feed_dict=retrieval_feed_dict_train)
        if step == 0:
            train_features = train_features_batch
        else:
            train_features = np.concatenate((train_features, train_features_batch), axis=0)

    for it in range(test_iteration + 1):
        if test_pre_index + batch_size_test < test_data_num:
            test_batch_x = test_x[test_pre_index: test_pre_index + batch_size_test]
        else:
            test_batch_x = test_x[test_pre_index:]
        test_pre_index += batch_size_test
        if np.shape(test_batch_x)[0] == 0:
            continue
        retrieval_feed_dict_test = {
            x: test_batch_x,
            training_flag: False
        }
        test_features_batch = sess.run(feature, feed_dict=retrieval_feed_dict_test)
        if it == 0:
            test_features = test_features_batch
        else:
            test_features = np.concatenate((test_features, test_features_batch), axis=0)

    gallery_x = train_features.astype(int)
    query_x = test_features

    Z_np = sess.run(Z)
    quantizedDist = pqDist(Z_np, n_book, gallery_x, query_x).T

    Rank = np.argsort(quantizedDist, axis=0)

    mAP = cat_apcal(label_Similarity, Rank, TOP_K)

    if visual_flag == True:

        rnd_id = np.random.randint(test_data_num, size=1)

        print("Visualize %d-th image"%(rnd_id))

        fig = plt.figure()
        query_to_show = test_x[rnd_id]
        query_to_show = color_deprocessing_toshow(query_to_show)
        query_ax = fig.add_subplot(2,8,1)
        query_ax.imshow(query_to_show)
        query_ax.set_title('Query', fontsize=10, color='m')
        query_ax.axis('off')
        plt.tight_layout()

        for i in range(2,17):
            gallery_to_show = db_x[Rank[i][rnd_id]]
            gallery_to_show = color_deprocessing_toshow(gallery_to_show)
            gallery_ax = fig.add_subplot(2,8,i)
            gallery_ax.imshow(gallery_to_show)
            gallery_ax.set_title('Rank %d'%(i-1), fontsize=10)
            gallery_ax.axis('off')
            plt.tight_layout()

        fig.tight_layout()
        plt.savefig('./retrieval_result_%d.png'%((rnd_id)))
        plt.close()

    return mAP
