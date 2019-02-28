import sys
sys.path.append('..')

from src.game import Game as PizzaGame
from trainer.env_manager import AIPlayer

import tensorflow as tf
from collections import deque
import numpy as np
import argparse
import os

ROWS = 12
COLS = 12
OBSERVATION_DIM = ROWS * COLS * 3 + 3
ACTIONS = ['up', 'down', 'left', 'right', 'toggle']
STATE_FEATURES = [
    'ingredients_map',
    'slices_map',
    'cursor_position',
    'slice_mode',
    'min_each_ingredient_per_slice',
    'max_ingredients_per_slice'
]

def gen_random_board(rows, cols):
    selection = ['M', 'T']
    return [''.join(np.random.choice(selection) for i in range(cols)) for j in range(rows)]

def main(args):
    board = gen_random_board(ROWS, COLS)
    policy_settings = {
        'n_actions': len(ACTIONS),
        'n_features': OBSERVATION_DIM,
        'learning_rate': args.learning_rate,
        'reward_decay': args.gamma,
        'output_graph': args.output_graph,
        'saving_model': args.saving_model,
        'policy_name': args.model_name
    }
    env_settings = {
        'actions': ACTIONS,
        'input_size': OBSERVATION_DIM,
        'render': args.render,
        'state_features': STATE_FEATURES,
    }
    pizza_config = {
        'pizza_lines': board,
        'r': ROWS,
        'c': COLS,
        'l': np.random.randint(1, 3),
        'h': np.random.randint(5, 10)
    }
    print(args.model_name)
    ai = AIPlayer(PizzaGame, env_settings, pizza_config, policy_settings)
    if args.saving_model:
        try:
            ai.policy.restore_model(args.model_name)
        except:
            print('No model found: ', model_name)
    for epc in range(args.n_epoch):
        for eps in range(args.n_episodes):
            print(f'Epoch: {epc} Game: {eps}')
            reward, score = ai.play_game(args.max_steps, epc)
            episode_score = reward + score
        ai.policy.learn(epc)
        #ai.policy.add_metrics(str(epc))
        ai.policy.clear_rollout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('pizza_ai trainer')
    parser.add_argument(
        '--n_epoch',
        type=int,
        default=10000)
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=200
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=ROWS * COLS * 10
    )
    parser.add_argument(
        '--restore',
        default=False,
        action='store_true')
    parser.add_argument(
        '--output_graph',
        default=True,
        action='store_true')
    parser.add_argument(
        '--render',
        default=False,
        action='store_true')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=.001)
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.95)
    parser.add_argument(
        '--saving_model',
        default=False,
        action='store_true')
    parser.add_argument(
        '--model_name',
        type=str,
        default='default')
    args = parser.parse_args()
    main(args)