import sys
sys.path.append('..')
import tensorflow as tf
import numpy as np
from policy_gradient import PolicyGradient
from src.game import Game




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

class EnvManager():

    def __init__(self, Game_class, init_config, env_settings):
        self.init_config = init_config
        self.Game = Game_class
        self.current_game = None
        self.game_memory = {}
        

        self.actions = env_settings['actions']

        self.policy = PolicyGradient(n_actions=len(self.actions), n_features=len(env_settings['state_features']))
        self.uniqueStateAccessor = {state_feature: Uniqifier() for state_feature in env_settings['state_features']}
        
    def reset(self,max_steps):
        self.current_game = self.Game({"max_steps": max_steps})

    def play_game(self):
        rewards = []
        self.reset(max_steps=100)
        obs, reward, done, info = self.start()
        obs = self.translate_observation(obs)
        while not self.current_game.env['done']:
            self.current_game.render()
            action = self.policy.choose_action(obs)
            obs, reward, done, info = self.current_game.step(self.actions[action])
            obs = self.translate_observation(obs)
            self.policy.store_transition(obs, action, reward)
            rewards.append(round(reward, 2))
        return sum(rewards), info['score']

    def translate_observation(self, obs):

        obs['ingredients_map'] = np.array(obs['ingredients_map']).flatten()
        obs['slices_map'] = np.array(obs['slices_map']).flatten()
        return np.array([
            self.uniqueStateAccessor['ingredients_map'].getIdFor(''.join(str(i) for i in obs['ingredients_map'])),
            self.uniqueStateAccessor['slices_map'].getIdFor(''.join(str(i) for i in obs['slices_map'])),
            self.uniqueStateAccessor['cursor_position'].getIdFor(obs['cursor_position']),
            1 if obs['slice_mode'] else 0,
            obs['min_each_ingredient_per_slice'],
            obs['max_ingredients_per_slice']
        ])    
    def start(self):
        return self.current_game.init(self.init_config)

def gen_random_board(rows, cols):
    selection = ['M', 'T']
    
    return [''.join(np.random.choice(selection) for i in range(cols)) for j in range(rows)]


board = gen_random_board(8, 8)



init_config = {
    'pizza_lines': board,
    'r': 8,
    'c': 8,
    'l': 1,
    'h': 6
}
max_steps = 100

env_settings = {
    'actions': ['up', 'down', 'left', 'right', 'toggle'],
    'state_features': [
        'ingredients_map',\
        'slices_map',\
        'cursor_position',\
        'slice_mode',\
        'min_each_ingredient_per_slice',\
        'max_ingredients_per_slice']

}

env = EnvManager(Game, init_config, env_settings)
episodes = 100
epochs = 5
avg_scores = []
avg_rewards = []

for epoch in range(epochs):
    rewards = [] 
    scores  = []
    for episode in range(episodes):
        print(f'Epoch: {epoch} => {episode}')
        reward, score = env.play_game()
        rewards.append(reward)
        scores.append(score)
    env.policy.learn()
    avg_rewards.append(sum(rewards) / len(rewards))
    avg_scores.append(sum(scores) / len(scores))

print(avg_rewards)
print(avg_scores)


