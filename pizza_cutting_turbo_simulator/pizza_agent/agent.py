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

        self.policy = PolicyGradient(n_actions=len(self.actions), n_features=len(env_settings['state_features']), name=env_settings['policy_name'])
        self.uniqueStateAccessor = {state_feature: Uniqifier() for state_feature in env_settings['state_features']}
        
    def reset(self,max_steps):
        self.current_game = self.Game({"max_steps": max_steps})

    def play_game(self, max_steps):
        rewards = []
        self.reset(max_steps)
        obs, reward, done, info = self.start()
        obs = self.translate_observation(obs)

        while not self.current_game.env['done']:
            # print(obs)
            action = self.policy.choose_action(obs)
            obs, reward, done, info = self.current_game.step(self.actions[action])
            obs = self.translate_observation(obs)
            self.policy.store_transition(obs, action, reward)
            rewards.append(round(reward, 2))
        self.current_game.render(self.init_config['r'], self.init_config['c'])
        # print (info)
        return sum(rewards), info['score']

    def translate_observation(self, obs):

        obs['ingredients_map'] = np.array(obs['ingredients_map']).flatten()
        obs['slices_map'] = np.array(obs['slices_map']).flatten()
        # obs['slices_map'] = sum(np.array(obs['slices_map']).flatten())
        return np.array([
            self.uniqueStateAccessor['ingredients_map'].getIdFor(''.join(str(i) for i in obs['ingredients_map'])),
            self.uniqueStateAccessor['slices_map'].getIdFor(''.join(str(i) for i in obs['slices_map'])) / 100,
            #self.uniqueStateAccessor['slices_map'].getIdFor(obs['slices_map']),
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


board = gen_random_board(12, 11)

init_config = {
    'pizza_lines': board,
    'r': 12,
    'c': 11,
    'l': 1,
    'h': 6
}
max_steps = 500

env_settings = {
    'actions': ['up', 'down', 'left', 'right', 'toggle'],
    'state_features': [
        'ingredients_map',\
        'slices_map',\
        'cursor_position',\
        'slice_mode',\
        'min_each_ingredient_per_slice',\
        'max_ingredients_per_slice'],
    'policy_name': 'default'
}


episode = {
    'count': 50,
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
    'count': 14,
    'scores': [],
    'rewards': [],
    'avg_rewards': 0,
    'avg_scores': 0,
    'max_score': 0
}

env = EnvManager(Game, init_config, env_settings)

for epc in range(epoch['count']):
    for eps in range(episode['count']):
        print(f'Epoch: {epc} Game: {eps}')
        reward, score = env.play_game(max_steps)
        episode_score = reward + score 
        episode['max_score'] = score if score > episode['max_score'] else episode['max_score']
        episode['max_reward'] = reward if reward > episode['max_reward'] else episode['max_reward']
        episode['rewards'].append(reward)
        episode['scores'].append(score)    
        #if episode_score > episode['max_combined_score_reward']:
        #    episode['trend'].append(episode_score)
        #    episode['max_combined_score_reward'] = episode_score
        #    env.policy.learn()
        #else:
        #    env.policy.clear_rollout()
    #Better
    env.policy.learn()
    epoch['rewards'].append(sum(episode['rewards']) / len(episode['rewards']))
    epoch['scores'].append(sum(episode['scores']) / len(episode['scores']))
    episode['rewards'] = [] 
    episode['scores'] = []
epoch['avg_rewards'] = sum(epoch['rewards']) / len(epoch['rewards'])
epoch['avg_scores'] = sum(epoch['scores']) / len(epoch['scores'])

print('epoch rewards: {}'.format(epoch['rewards']))
print('epoch scores: {}'.format(epoch['scores']))
print('reward/score trend: ', episode['trend'])
print('Average rewards over all epochs: ', epoch['avg_rewards'])

for key, value in env.uniqueStateAccessor.items():
    print(value.id)