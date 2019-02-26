import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential
import numpy as np
import time

def get_input_data():
    return np.zeros(3, dtype=int32), np.ones(432, dtype=int32)


def build_network(x1_nfeatures, x2_nfeatures):

    with tf.name_scope('inputs'):
        x1 = tf.placeholder(tf.int32, [None, x1_nfeatures], name='board-layout')
        x2 = tf.placeholder(tf.int32, [None, x2_nfeatures], name='context')

    model.add(Dense(10, input_shape=())) 



x1, x2 = get_input_data()

build_network(len(x1), len(x2))


#inputs = Input((2, ))
#layer = Dense(10)
#x  = layer(inputs)
#layer.trainable=True
#time.sleep(5)
#print(layer.get_weights())


