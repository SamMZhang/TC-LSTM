# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sqlite3
from sklearn import preprocessing
from scipy import stats
import scipy.io as sio

mean = 0
std = 0


def one_hot_en(dat):
    enc = preprocessing.OneHotEncoder()
    enc.fit([i+1 for i in range(24)])

    return enc.transform(dat)


def normalized_views(data_all):
    mean1 = np.mean(data_all[:, 1:], axis=0)
    std1 = np.std(data_all[:, 1:], axis=0)
    # only normalized the review data regardless of timestamp data
    normalized_data_view = (data_all[:, 1:] - np.mean(data_all[:, 1:], axis=0)) / np.std(data_all[:, 1:], axis=0)
    # (timestamp, nor_view, nor_groudt)
    normalized_data = np.vstack((data_all[:, 0], normalized_data_view[:, 0], normalized_data_view[:, 1]))
    return mean1, std1, normalized_data


def lstmCell():
    # basicLstm??
    basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # dropout
    drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
    return basicLstm


def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # reshape to 2-D
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # tensor=3-D, As the input of lstm cell
    # cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
    # Multi layer LSTM stack
    # cell_lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    # number_of_layers = 2
    # cell = tf.nn.rnn_cell.MultiRNNCell([cell_lstm] * number_of_layers)
    ###
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states


# ?????
def get_train_data(batch_size, time_step, train_begin, train_end):
    batch_index = []
    data_train = data[train_begin:train_end]
    # ???views
    global mean
    global std
    mean, std, normalized_train_data = normalized_views(data_train)
    normalized_train_data = np.transpose(normalized_train_data)
    train_x, train_y = [], []  # ???
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i:i + time_step, :2]
        y = normalized_train_data[i + time_step, -1, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))

    return batch_index, train_x, train_y


# ?????
def get_test_data(time_step, test_begin):
    data_test = data[test_begin:]
    mean1, std1, normalized_test_data = normalized_views(data_test)  # normalization, mean1 & std1 are useless
    normalized_test_data = np.transpose(normalized_test_data)
    size = (len(normalized_test_data) + time_step) // time_step  # size samples
    test_x, test_y = [], []
    for i in range(len(normalized_test_data) - time_step):
        x = normalized_test_data[i:i + time_step, :2]
        y = normalized_test_data[i + time_step, -1]

    test_x.append((normalized_test_data[(i + 1) * time_step:, :2]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, -1]).tolist())

    return test_x, test_y


def train_lstm(batch_size, time_step, train_begin, train_end):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)
    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

    with tf.Session() as sess:
        loss_ = 0
        sess.run(tf.global_variables_initializer())
        for i in range(50):  # ?????????????????????????????
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            print("Number of iterations:", i, " loss:", loss_)
        print("model_save: ", saver.save(sess, 'model_save2/modle.ckpt'))
        # I run the code on windows 10,so use  'model_save2\\modle.ckpt'
        # if you run it on Linux,please use  'model_save2/modle.ckpt'
        print("The train has finished")


def prediction(time_step, test_begin):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    test_x, test_y = get_test_data(time_step=time_step, test_begin=test_begin)
    global mean
    global std
    with tf.variable_scope("sec_lstm", reuse=True):
        pred, _ = lstm(X)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        # ????
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict = []
        for step in range(len(test_x) - 1):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[-1] + mean[-1]
        test_predict = np.array(test_predict) * std[-1] + mean[-1]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]))  # MAE
        # ave_test_predict = test_predict
        # for i in range(len(ave_test_predict)):
        #     ave_test_predict[i] = np.mean(test_y)
        # src = stats.spearmanr(ave_test_predict, test_y[:len(test_predict)])  # spearman ranking correlation
        src = stats.spearmanr(test_predict, test_y[:len(test_predict)])  # spearman ranking correlation
        print("The MAE/SRC of this predict:", acc, src)
        # ????????
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b', label='preditc result')
        plt.plot(list(range(len(test_y))), test_y, color='r', label='Y - ground-truth')
        plt.legend(loc='upper right')
        plt.show()


def read_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    conn.row_factory = sqlite3.Row
    sql = """select views from 'tempor' order by video_id"""
    cursor.execute(sql)
    views = cursor.fetchall()
    sql = """select distinct video_id from 'tempor'"""
    cursor.execute(sql)
    video_id = cursor.fetchall()

    n = 1  # number of user
    start_usr = (n - 1) * int(len(views) / len(video_id))
    end_usr = int(len(views) / len(video_id))  # review_count for one video = 2058
    # generate reviews
    review = views[:end_usr]
    review = review[:288]  # one-day and one video's review sequence
    review = np.array([i[0] for i in review])  # tuple to float
    review = np.hstack((review[0], review[1:] - review[0:-1]))  # sum to timely popularity

    # one-hot temporal encode
    sql = """select time from 'tempor' order by video_id"""
    cursor.execute(sql)
    time_all = cursor.fetchall()  # one timestamp
    time_start = time_all[0:288]
    hour = [int(i[0][11:13]) for i in time_start]  # timestamp for hour

    # Topic = one_hot_en(hour)
    Topic = hour

    data = np.vstack((Topic, review, np.hstack((review[1:], review[-1]))))  # (review, topic, Ground-Truth)

    return data


if __name__ == '__main__':
    rnn_unit = 10  # unit numbers of hidden layers
    lstm_layers = 2  # numbers of hidden layers
    input_size = 2  # input_size
    output_size = 1
    lr = 0.0006  # learning rate

    # path_db = '/home/zzb/Data/Xigua_new/200.111new_finish/xigua29.db'
    # data = np.transpose(read_db(path_db))
    datas = sio.loadmat('../../DataPre/videoid_review_timestamp_-1-1_5%_289.mat')
    video_id = datas['video_ids'][0]
    reviews = datas['videoid_reviews_0_0'][0]
    timestamp = datas['videoid_timestamp_0'][0]

    # fea = data1['fea_co_norm']
    # data1 = sio.loadmat('../../FeatureExtract/multi_modal_features_co_norm.mat')

    data = np.vstack((timestamp, reviews, np.hstack((reviews[1:], reviews[-1])))).T


    weights = {
        'in': tf.Variable(tf.random_normal([input_size, rnn_unit], seed=1)),
        'out': tf.Variable(tf.random_normal([rnn_unit, 1], seed=1))
    }
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[rnn_unit, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    train_lstm(batch_size=24, time_step=4, train_begin=0, train_end=192)

    prediction(time_step=4, test_begin=192)
