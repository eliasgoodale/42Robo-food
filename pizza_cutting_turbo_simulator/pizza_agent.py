from src.game import Game
import numpy as np
import random
import tensorflow as tf
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam

config = {
    'board'
}


class PizzaEnv(gym.Env):

    def __init__(self, config):
        
    def reset(goal_steps=100):
        default_args = {'max_steps': goal_steps}
        pizza_lines = ['TTTTT', 'TMMMT', 'TTTTT']
        pizza_config = { 'pizza_lines': pizza_lines, 'r': 3 , 'c': 5, 'l': 1, 'h': 6 }

        game = Game(default_args)
        state, reward, done, info = game.init(pizza_config)

        return game, state, reward, done, info

    def generate_episode():
        episode = []
        game, state, reward, done, info = reset()

        while not done:
            action = get_action_policy()
