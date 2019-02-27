
import subprocess as sp
import sys
import select
sys.path.append('..')
import tensorflow as tf
import numpy as np
from policy_gradient import PolicyGradient
from src.game import Game
import time

class Uniqifier(object):
    def __init__(self):
        self.id = 1
        self.element_map = {}
        self.reverse_map = {}

    def getIdFor(self, obj):
        obj_id = self.element_map.get(obj)
        if obj_id is None:
            obj_id = self.id
            self.element_map[obj] = obj_id
            self.reverse_map[obj_id] = obj
            self.id += 1
        return obj_id

    def getObj(self, id):
        return self.reverse_map.get(id)

class GameViewer:
    def __init__(self, render_fn, fps, rendering=False):
        self.render_fn = render_fn
        self.fps = fps
        self.rendering=rendering

    def render(self):
        time.sleep(self.fps)
        sp.call('clear',shell=True)
        self.render_fn()

class EnvManager():

    def __init__(self, Game_class, init_config, env_settings):
        self.init_config = init_config
        self.Game = Game_class
        self.current_game = None
        self.game_memory = {}
        self.prev_cursor_pos = (0, 0)
        self.actions = env_settings['actions']
        self.policy = PolicyGradient(n_actions=len(self.actions), n_features=env_settings['input_size'], name=env_settings['policy_name'])
        self.uniqueStateAccessor = {state_feature: Uniqifier() for state_feature in env_settings['state_features']}

    def reset(self,max_steps):
        self.current_game = self.Game({"max_steps": max_steps})
        self.game_viewer = GameViewer(self.current_game.render_live, 1/10)

    def check_cursor_progression(self,cursor):
        reward_bonus = np.sum(np.subtract(self.prev_cursor_pos, cursor))
        #print(abs(reward_bonus), self.prev_cursor_pos, cursor)
        return abs(reward_bonus)


    def play_game(self, max_steps, epc):
        rewards = []
        self.reset(max_steps)
        obs, reward, done, info = self.start()
        obs = self.feed_observation(obs)
        #print(obs, len(obs))
        while not self.current_game.env['done']:
            action = self.policy.choose_action(obs)
            obs, reward, done, info = self.current_game.step(self.actions[action])

            #if obs['slice_mode']:
            #    reward += self.check_cursor_progression(obs['cursor_position'])
            #    self.prev_cursor_pos = obs['cursor_position']

            obs = self.feed_observation(obs)
            if (self.game_viewer.rendering):
                self.game_viewer.render()
            self.policy.store_transition(obs, action, reward)
            rewards.append(round(reward, 2))
        self.current_game.render()
        # print('EVERYDAY IM SHUFFLIN: ', ''.join(['>' for i in range(self.current_game.google_engineer.shuffle_count)]))
        # print('TOGGLE_BOOOII: ', ''.join(['>' for i in range(self.current_game.google_engineer.double_toggle_count)]))
        # print (info)
        return sum(rewards), info['score']

    def feed_observation(self, obs):
        #print(obs)
        board = np.array(obs['ingredients_map']).flatten()

        slices_map = np.array(obs['slices_map']).flatten()

        cursor = np.zeros(len(board), dtype=int)
        pos = obs['cursor_position'][0] + obs['cursor_position'][1]
        cursor[pos] = 1

        #print ('board: ', board)
        #print ('slices_map: ', slices_map)
        #print ('cursor: ', cursor)
        slice_mode = 1 if obs['slice_mode'] else 0
        min_ingred = obs['min_each_ingredient_per_slice']
        max_ingred = obs['max_ingredients_per_slice']
        #print("slice_mode: ", slice_mode)
        #print("min_ingred: ", min_ingred)
        #print("max_ingred: ", max_ingred)
        context = np.array([slice_mode, min_ingred, max_ingred])
        layout = np.concatenate((board, slices_map, cursor))

        inputs = np.concatenate((context, layout))

        return inputs

    def start(self):
        return self.current_game.init(self.init_config)

