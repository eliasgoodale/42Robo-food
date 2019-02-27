import sys
sys.path.append('..')
import argparse
import os
import tensorflow as tf
import numpy as np
from policy_gradient import PolicyGradient
from src.game import Game
from collections import deque


def gen_random_board(rows, cols):
    selection = ['M', 'T']
    return [''.join(np.random.choice(selection) for i in range(cols)) for j in range(rows)]

ROWS = 12
COLS = 12
OBSERVATION_DIM = ROWS * COLS * 3 + 3

ACTIONS = ['up', 'down', 'left', 'right', 'toggle']

MEMORY_CAPACITY = 1000000
ROLLOUT_SIZE = 10000

MEMORY = deque([], maxlen=MEMORY_CAPACITY)

def build_graph(self):

    
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        layer_h1 = tf.layers.dense(
            inputs=tf_obs,
            units=42,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        layer_h2 = tf.layers.dense(
            inputs=layer_h1,
            units=21,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        logits = tf.layers.dense(
            inputs=layer_h2,
            units=len(ACTIONS),
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc3'
        )
    return logits

        all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob)*tf.one_hot(tf_acts, n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * tf_vt)  # reward guided loss
            # IAN ADD
            loss_sum = tf.summary.scalar(name='loss summary per epoch', tensor=loss)

        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss)

def gen():
    for m in list(MEMORY):
        yield m

def main(args):
    args_dict = vars(args)
    print('args: {}'.format(args_dict))

    with tf.Graph().as_default() as g:
        
        with tf.name_scope('rollout'):
            observations = tf.placeholder(shape=(None, OBSERVATION_DIM), dtype=tf.int32)
            logits = build_graph(observations)
            logits_for_sampling = tf.reshape
            sample_action = tf.squeeze(tf.multinomial(logits=logits_for_sampling, num_samples = 1))

        optimizer = tf.train.AdamOptimizer(
            learning_rate=args.learning_rate,
            decay=args.decay
        )

        with tf.name_scope('dataset'):
            ds = tf.data.Dataset.from_generator(gen, output_types={tf.float32, tf.float32, tf.float32})
            ds = ds.shuffle(MEMORY_CAPACITY).repeat().batch(args.batch_size)
            iterator = ds.make_one_shot_iterator()
        
        with tf.name_scope('train'):
            # ADD RETURN VALUE FROM GAME HERE
            next_batch = iterator.get_next()
            train_observations, labels, processed_rewards = next_batch

            train_observations.set_shape((args.batch_size OBSERVATION_DIM))
            train_logits = build_graph(train_observations)

            cross_entropies = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=train_logits,
                labels=labels
            )
            probs = tf.nn.softmax(logits=train_logits)
            move_cost = args.laziness * tf.reduce_sum(probs * [0, 1.0, 1.0], axis=1)
            loss = tf.reduce_sum(procesed_rewards * cross_entropies + move_cost)

            global_step = tf.train.get_or_create_global_step()
            train_op = optimizer.minimize(loss, global_step=global_step)
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=args.max_to_keep)
        
        with tf.name_scope('summaries'):
            rollout_reward = tf.placeholder(
                shape=(),
                dtype=tf.float32
            )
            hidden_weights = tf.trainable_variables()[0]
            for h in range(args.hidden_dim):
                slice_ = tf.slice(hidden_weights, [0, h], [-1, 1])
                image = tf.reshape(slice_, [1, 80, 80, 1])
                tf.summary.image('hidden_{:04d}'.format(h), image)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
                tf.summary.scalar('{}_max'.format(var.op.name), tf.reduce_max(var))
                tf.summary.scalar('{}_min'.format(var.op.name), tf.reduce_min(var))
            
            tf.summary.scalar('rollout_reward', rollout_reward)
            tf.summar.scalar('loss', loss)

            merged= tf.summary.merge_all()
        print('Number of trainable variables: {}'.format(len(tf.trainable_variables())))


        board = gen_random_board(ROWS, COLS)
        env_settings = {
            'actions': ACTIONS,
            'input_size': OBSERVATION_DIM,
            'state_features': [
                'ingredients_map',
                'slices_map',
                'cursor_position',
                'slice_mode',
                'min_each_ingredient_per_slice',
                'max_ingredients_per_slice'],
            'policy_name': 'foodie-do'
        }
        init_config = {
            'pizza_lines': board,
            'r': ROWS,
            'c': COLS,
            'l': np.random.randint(1, 3),
            'h': np.random.randint(5, 10)
        }
        env = EnvManager(Game, env_settings, init_config)

        with tf.Session(graph=g) as sess:
            if args.restore:
                restore_path = tf.train.latest_checkpoint(args.output_dir)
                print('Restoring from {}'.format(restore_path))
                saver.restore(sess, restore_path)
            else:
                sess.run(init)
            summary_path = os.path.join(args.output_dir, 'summary')
            summary_writer = tf.summary.FileWriter(summary_path, sess.graph)

            _rollout_reward = -100.0

            for i in range(args.n_epoch):
                
                print('>>>>>>> epoch {}'.format(i+1))
                print('>>> Rollout phase')
                env.reset()
                epoch_memory = []
                episode_memory = []

                _observation = np.zeros(OBSERVATION_DIM)

                while True:
                    _label = sess.run(sample_action, feed_dict={observations: [_observation]})

                    _action = ACTIONS[_label]

                    _pair_state, _reward, _done



https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rl-pong/trainer/task.py


    with tf.name_scope('inputs'):
        tf_obs = tf.placeholder(tf.float32, [None, OBSERVATION_DIM])
        tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")




def gen_random_board(rows, cols):
    selection = ['M', 'T']
    return [''.join(np.random.choice(selection) for i in range(cols)) for j in range(rows)]
ROWS = 12
COLS = 12

board = gen_random_board(ROWS, COLS)

init_config = {
    'pizza_lines': board,
    'r': ROWS,
    'c': COLS,
    'l': np.random.randint(1, 3),
    'h': np.random.randint(5, 10)
}
max_steps = ROWS * COLS * 10

env_settings = {
    'actions': ACTIONS,
    'input_size': ROWS * COLS * 3 + 3,
    'state_features': [
        'ingredients_map',\
        'slices_map',\
        'cursor_position',\
        'slice_mode',\
        'min_each_ingredient_per_slice',\
        'max_ingredients_per_slice'],
    'policy_name': 'foodie-do'
}


episode = {
    'count': 100,
    'scores': [],
    'rewards': [],
    'avg_rewards': 0,
    'avg_scores': 0,
    'max_score': 0,
    'max_reward': 0,
    'trend': [],
    'max_combined_score_reward': 0
}

epoch = {
    'count': 10,
    'scores': [],
    'rewards': [],
    'avg_rewards': 0,
    'avg_scores': 0,
    'max_score': 0
}

env = EnvManager(Game, init_config, env_settings)
try:
    model_name = input("what module do you want to restore: ")
    env.policy.restore_model(model_name)
except:
    input("no model to restore.")
for epc in range(epoch['count']):
    for eps in range(episode['count']):
        print(f'Epoch: {epc} Game: {eps}')
        reward, score = env.play_game(max_steps, epc)
        episode_score = reward + score
        episode['max_score'] = score if score > episode['max_score'] else episode['max_score']
        episode['max_reward'] = reward if reward > episode['max_reward'] else episode['max_reward']
        episode['rewards'].append(reward)
        episode['scores'].append(score)
    # IAN ADD, pass epc to env.policy.learn
    env.policy.learn(epc)
    env.policy.add_metrics(str(epc))
    env.policy.clear_rollout()
        #if episode_score > episode['max_combined_score_reward']:
        #    episode['trend'].append(episode_score)
        #    episode['max_combined_score_reward'] = episode_score
        #    env.policy.learn()
        #else:
        #    env.policy.clear_rollout()
    #Better


    epoch['rewards'].append(sum(episode['rewards']) / len(episode['rewards']))
    epoch['scores'].append(sum(episode['scores']) / len(episode['scores']))
    episode['rewards'] = []
    episode['scores'] = []
epoch['avg_rewards'] = sum(epoch['rewards']) / len(epoch['rewards'])
epoch['avg_scores'] = sum(epoch['scores']) / len(epoch['scores'])

print('epoch rewards: ', epoch['rewards'])
print('epoch scores: ', epoch['scores'])
print('reward/score trend: ', episode['trend'])
print('Average rewards over all epochs: ', epoch['avg_rewards'])
