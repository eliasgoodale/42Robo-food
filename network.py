import tensorflow as tf

ROWS = 4
COLS = 4

learning_rate = 0.1

n_hidden_1 = 10

with tf.name_scope('Inputs'):
    X1 = tf.placeholder(tf.int32, shape=(ROWS, COLS), name="board")
    X2 = tf.placeholder(tf.int32, shape=(ROWS, COLS), name="slices")
    X3 = tf.placeholder(tf.int32, shape=(1, 2), name="cursor")
    X4 = tf.placeholder(tf.bool, shape=(), name="slice_mode_on")
    X5 = tf.placeholder(tf.int32, shape=(), name="min_ingred")
    X6 = tf.placeholder(tf.int32, shape=(), name="max_slice")