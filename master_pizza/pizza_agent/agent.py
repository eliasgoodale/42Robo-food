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
        self.prev_cursor_pos = (0, 0)
        self.actions = env_settings['actions']
        self.policy = PolicyGradient(n_actions=len(self.actions), n_features=env_settings['input_size'], name=env_settings['policy_name'])
        self.uniqueStateAccessor = {state_feature: Uniqifier() for state_feature in env_settings['state_features']}
        
    def reset(self,max_steps):
        self.current_game = self.Game({"max_steps": max_steps})
    
    def check_cursor_progression(self,cursor):
        reward_bonus = np.sum(np.subtract(self.prev_cursor_pos, cursor))
        #print(abs(reward_bonus), self.prev_cursor_pos, cursor)
        return abs(reward_bonus)


    def play_game(self, max_steps):
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

            self.policy.store_transition(obs, action, reward)
            rewards.append(round(reward, 2))
        self.current_game.render()
        #print('EVERYDAY IM SHUFFLIN: ', ''.join(['>' for i in range(self.current_game.google_engineer.shuffle_count)]))
        #print('TOGGLE_BOOOII: ', ''.join(['>' for i in range(self.current_game.google_engineer.double_toggle_count)]))
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
    'l': 1,
    'h': 6
}
max_steps = 500

env_settings = {
    'actions': ['down', 'right', 'toggle'],
    'input_size': 435,
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

for epc in range(epoch['count']):
    for eps in range(episode['count']):
        print(f'Epoch: {epc} Game: {eps}')
        reward, score = env.play_game(max_steps)
        episode_score = reward + score 
        episode['max_score'] = score if score > episode['max_score'] else episode['max_score']
        episode['max_reward'] = reward if reward > episode['max_reward'] else episode['max_reward']
        episode['rewards'].append(reward)
        episode['scores'].append(score)
        env.policy.learn()
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





