
import tensorflow as tf
import random
import numpy as np

ROWS = 3
COLS = 5
NUM_INGRED = 2

ACTIONS = ['up', 'down', 'left', 'right', 'toggle']

n_classes = len(ACTIONS)
learning_rate = 0.1

n_hidden_1 = 60


n_input_1 = ROWS * COLS
n_input_2 = 3

with tf.name_scope('X1'):
    x1 = tf.placeholder(tf.int32, shape=(1, n_input_1), name="board") # [0, ... n] values(1, 0)
    x2 = tf.placeholder(tf.int32, shape=(1, n_input_1), name="slices")# [0 ... n] values(-1, 0, ... slicemax)
    x3 = tf.placeholder(tf.int32, shape=(1, n_input_1), name="cursor")# [0 ... n] values )(1, 0 | if 1 all 0)
    X1 = tf.concat([x1, x2, x3], axis=1) #X1 [0 ... 3n] 
#dynamic_shape = tf.shape(X1)

with tf.name_scope('X2'):
    x4 = tf.placeholder(tf.int32, shape=(), name="slice_mode_on")
    x5 = tf.placeholder(tf.int32, shape=(), name="min_ingred")
    x6 = tf.placeholder(tf.int32, shape=(), name="max_slice")
    X2 = tf.stack([x4, x5, x6], axis=0)
#print (X2.get_shape())


with tf.name_scope('Outputs'):
    Y = tf.placeholder('float', [1, n_classes], name='p_action')




def build_network(X, W, B):
    with tf.name_scope('Hidden'):
        h1_x1w1 = tf.matmul(X['1'], W['1']['h1'])
        h1_x2w2 = tf.matmull(X['2'], W['2']['h1'])
        
        h1 = tf.sigmoid(tf.add(tf.matmul(X['1'], weights['h1'], biases['b1'])))
        out = tf.nn.softmax(tf.add(tf.matmul(h1, weights['out']), biases['out']))
    return out

X: {
    '1': X1
    '2': X2
}

W = {
    '1': {
        'h1': tf.Variable(tf.random_normal([n_input_1, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
    '2': {
        'h1': tf.Variable(tf.random_normal([n_input_2, n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
    }
}

B = {
    '1': {
        'h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))

    }
    '2': {
        'h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
}




#predict = multilayer_network(X1, weights, biases)
#cost = tf.contrib.losses.softmax_cross_entropy(predict, Y)
#optimizer = tf.train.AdamOptimizer().minimize(cost)

arr = np.random.randint(8, size=(1, np.random.randint(low=1, high=1000000)))
with tf.Session() as sess:
    tf.summary.FileWriter("logs/", sess.graph)



    

