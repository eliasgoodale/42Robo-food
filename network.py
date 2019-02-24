import tensorflow as tf

ROWS = 3
COLS = 5
NUM_INGRED = 2

ACTIONS = ['up', 'down', 'left', 'right', 'toggle']

n_classes = len(ACTIONS)
learning_rate = 0.1

n_hidden_1 = 60
n_input = ROWS * COLS * NUM_INGRED

with tf.name_scope('Inputs'):
    x1 = tf.placeholder(tf.int32, shape=(ROWS, COLS), name="board")
    x2 = tf.placeholder(tf.int32, shape=(ROWS, COLS), name="slices")
    x3 = tf.placeholder(tf.int32, shape=(1, 2), name="cursor")
    x4 = tf.placeholder(tf.int32, shape=(), name="slice_mode_on")
    x5 = tf.placeholder(tf.int32, shape=(), name="min_ingred")
    x6 = tf.placeholder(tf.int32, shape=(), name="max_slice")
    X = tf.reshape(tf.concat(0,[x1, x2, x3, x4, x5, x6]), [1, n_input])

with tf.name_scope('Outputs'):
    Y = tf.placeholder(tf.float, [1, n_classes], name='p_action')




def multilayer_network(x, weights, biases):
    h1 = tf.signmoid(tf.add(tf.matmul(x, weights['h1'], biases['b1'])))
    out = tf.nn.softmax(tf.add(tf.matmul(h1, weights['out']), biases['out']))
    return out

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal(n_hidden_1, n_classes))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

predict = multilayer_network(X, weights, biases)
cost = tf.contrib.losses.softmax_cross_entropy(predict, Y)
optimizer = tf.train.AdamOptimizer().minimize(cost)


    

