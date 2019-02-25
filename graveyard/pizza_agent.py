from pizza_cutting_turbo_simulator.src.game import Game
import numpy as np
import random
import tensorflow as tf
from keras.models     import Sequential
from keras.layers     import Dense
from keras.optimizers import Adam
'''

Game loop with random actions

while not game.env['done']:
    action = random.choice(actionspace)
    print(action, type(action))
    game.step(action)
    game.render()

'''

class ItemUniqifier(object):
    def __init__(self):
        self.id = 0
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

uniqueEngine = ItemUniqifier()


def reset(goal_steps=100):
    default_args = {'max_steps': goal_steps}
    pizza_lines = ['TTTTT', 'TMMMT', 'TTTTT']
    pizza_config = { 'pizza_lines': pizza_lines, 'r': 3 , 'c': 5, 'l': 1, 'h': 6 }
    
    game = Game(default_args)
    game.init(pizza_config)
    
    return game

def format_observation(obs):
    return [uniqueEngine.getIdFor(''.join('{}{}'.format(key, val) for key, val in obs.items()))]

#def format_observation(obs):
#
#    return np.array([
#        uniqueEngine.getIdFor(tuple(map(tuple, obs['ingredients_map']))), 
#        uniqueEngine.getIdFor(tuple(map(tuple, obs['slices_map']))),
#        uniqueEngine.getIdFor(obs['cursor_position']), 
#        uniqueEngine.getIdFor(obs['slice_mode']), 
#        uniqueEngine.getIdFor(obs['min_each_ingredient_per_slice']),
#        uniqueEngine.getIdFor(obs['max_ingredients_per_slice'])])

actionspace = ['up', 'down', 'left', 'right', 'toggle']
#action_index = [0, 1, 2, 3 ,4]
#action_depth = 5
#actionspace_onehot = tf.one_hot(action_index, action_depth)


score_requirement = 15
initial_games = 1000
r_table = np.zeros((100000, 5))

def model_data_preparation(goal_steps=500):
    env = reset(goal_steps)
    training_data = []
    accepted_scores = []
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        while not env.env['done']:
            #env.render()
            action = random.choice(actionspace)
            action_index = actionspace.index(action)
            observation, reward, done, info = env.step(action)
            print(observation)
            #observation = format_observation(observation)
            #r_table[observation[0], action_index] = reward
            #print(observation[0])
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
            
            previous_observation = observation
            score = info['score']

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 'up':
                    output = 0
                    #output = [1, 0, 0, 0, 0]
                elif data[1] == 'down':
                    output = 1
                    #output = [0, 1, 0, 0, 0]                
                elif data[1] == 'left':
                    output = 2
                    #output = [0, 0, 1, 0, 0]
                elif data[1] == 'right':
                    output = 3
                    #output = [0, 0, 0, 1, 0]
                elif data[1] == 'toggle':
                    output = 4
                    #output = [0, 0, 0, 0, 1]
                training_data.append([data[0][0], output])
                #print(training_data)
        env = reset()
    #print(accepted_scores)
    return training_data

def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam())
    return model

def train_model(training_data):
    X = np.array([data[0] for data in training_data])
    Y = np.array([data[1] for data in training_data])
    model = build_model(input_size=len(X[0]), output_size=len(Y[0]))
    model.fit(X, Y, epochs=5)
    return model

training_data = model_data_preparation()

#print(training_data[0])
#print(r_table)
'''
env = reset()
tot_reward = 0
s = 0
while not env.env['done']:
    print("LOOP",s)
    a = np.argmax(r_table[s, :])
    s, r, done, info = env.step(actionspace[a])
    s = format_observation(s)
    tot_reward += r

print(tot_reward)


trained_model = train_model(training_data)

scores = []
choices = []

for episode in range(1):
    env = reset()
    score = 0
    prev_obs = np.array([])
    while not env.env['done']:
        env.render()
        if len(prev_obs) == 0:
            action = random.choice(actionspace)
        else:
            action = actionspace[np.argmax(trained_model.predict(prev_obs)[0])]
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        new_observation = format_observation(new_observation)
        prev_obs = new_observation
        score = info['score']
        #print(action)
        if done:
            break
    env = reset()
    scores.append(score)

print(scores)
print('Average Score: ', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
'''



