import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
'''
Board initial state:
ingredients_map  =>  [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
ingredients_map  :  <class 'list'>
slices_map  =>  [[-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1], [-1, -1, -1, -1, -1]]
slices_map  :  <class 'list'>
cursor_position  =>  (0, 0)
cursor_position  :  <class 'tuple'>
slice_mode  =>  False
slice_mode  :  <class 'bool'>
min_each_ingredient_per_slice  =>  1
min_each_ingredient_per_slice  :  <class 'int'>
max_ingredients_per_slice  =>  6
max_ingredients_per_slice  :  <class 'int'>

Pizza Initializer:
pizza_lines  =>  ['TTTTT', 'TMMMT', 'TTTTT']
pizza_lines  :  <class 'list'>
r  =>  3 <class 'int'>
c  =>  5 <class 'int'>
l  =>  1 <class 'int'>
h  =>  6 <class 'int'>
min_each_ingredients 1 <class 'int'>
max_ingredients_per_slice 6 <class 'int'>
cursor_position (0, 0) <class 'tuple'>
slice_mode False <class 'bool'>
valid_slices [] <class 'list'>
score 0 <class 'int'>
'''

# Holds all information and logic for environment
class EnvManager:

    def __init__(self, options):
        self.ingredients = options['ingredients']
        self.board = options['board']

    def initial_state(self):
        return {
            'ingredients_map': [[random.choice(list(self.ingredients.values()))\
                for i in range(self.board["columns"])] for j in range(self.board['rows'])],
            
            'slices_map': np.full((self.board['rows'], self.board['columns']), -1).tolist(),
            'cursor_position': (0, 0),
            'slice_mode': False,
            'min_each_ingredient_per_slice': 1,
            'max_ingredients_per_slice': 6,
        }
# Has methods that interact with the environment
class PizzaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_manager):
        self.env_manager = env_manager
        self.max_steps = 100
        self.step_index = 0
        self.env = {
            "state": self.env_manager.initial_state(),
            "reward": 0,
            "done": False,
            "information": {
                'step': self.step_index,
                'action': 'none',
                'unique_ingredients': 2,
            }
        }
    def init_state():
           return self.env['state'], self.env['reward'],  
        


    
    #def step(self, action):

    #def reset(self):

    #def render(self, mode='human', close=False):

env_options = {
    'board': {'rows': 2, 'columns': 5},
    'ingredients': {'M': 0, 'T': 0}
}
env_manager = EnvManager(env_options)

env = PizzaEnv(env_manager)

print(env.env['state'])
#print (env.env_manager.ingredients.values().tolist(), type(env.env_manager.ingredients.values()))
#for key, value in env.env["state"].items():
#    print(f'{key} => {value}')