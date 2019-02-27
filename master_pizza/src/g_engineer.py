from src.pizza import Pizza, Direction

import numpy as np
import json

class GoogleEngineer:
    delta_position = {
        Direction.up: (-1, 0),
        Direction.down: (1, 0),
        Direction.left: (0, -1),
        Direction.right: (0, 1),
        'toggle': 'toggle',
    }

    reward_context = {
            'up': 'move',
            'down': 'move',
            'left': 'move',
            'right': 'move',
            'toggle': 'toggle'
    }

    rewards = {
        'POSITIVE': {
            'MADE_SQUARE': 1.0,
            'MOVED_OUT_OF_SLICE': 0.1,
            'TOGGLED_ON_FIRST_ACTION_OUT_OF_SLICE': 1.0
        },
        
        'NEGATIVE': {
            'MOVED_OUT_OF_BOUNDS': -1.0,
            'DID_NOT_MOVE_OUT_OF_SLICE': -0.1,
            'BACKTRACK': -1.0,
            'DOUBLE_TOGGLE': -1.0
        },

        'NEUTRAL': {
            'MOVED': 0.0,
            'SUB_OPTIMAL_SLICE': 0.0
        }
    }
    def __init__(self, pizza_config):

        self.pizza = Pizza(pizza_config['pizza_lines'])
        self.min_each_ingredient_per_slice = pizza_config['l']
        self.max_ingredients_per_slice = pizza_config['h']
        
        # Game memory
        self.prev_action = 'none'
        self.curr_action = 'none'
        self.next_action = 'none'
        self.current_game_step = 0
        
        self.previous_position = (-1, -1)
        self.cursor_position = (0, 0)
        self.next_cursor_position = (-1, -1)

        # Punishment action counts
        self.backtrack_count = 0
        self.double_toggle_count = 0

        # Positional Data

        self.slice_mode = False
        self.valid_slices = []
        self.score = 0
    

    def score_of(self, slice):
        if min(self.pizza.ingredients.of(slice)) >= self.min_each_ingredient_per_slice:
            return slice.ingredients
        return 0
    
    def movement_in_bounds(self):
        y, x = self.next_cursor_position
        return (0 <= y < self.pizza.r) and (0 <= x < self.pizza.c)

    def movement(self):
        self.previous_position = self.cursor_position
        self.cursor_position = self.next_cursor_position
        return self.rewards['NEUTRAL']['MOVED']
    
    def did_backtrack(self):
        #print ('prev: ', self.previous_position, 'curr: ', self.cursor_position, 'next: ',self.next_cursor_position)
        if self.previous_position == self.next_cursor_position:
            self.backtrack_count+=1
            return self.rewards['NEGATIVE']['BACKTRACK']
        else:
            return 0

    def moved_out_of_slice(self):
        slice_map = self.pizza._map
        dx = self.next_cursor_position[0]
        dy = self.next_cursor_position[1]
        x = self.cursor_position[0]
        y = self.cursor_position[1]

        if slice_map[x][y] != -1 and slice_map[dx][dy] == -1:
            return self.rewards['POSITIVE']['MOVED_OUT_OF_SLICE']
        elif slice_map[x][y] != -1 and slice_map[dx][dy] != -1:
            return self.rewards['NEGATIVE']['DID_NOT_MOVE_OUT_OF_SLICE']
        else:
            return self.rewards['NEUTRAL']['MOVED']


    def increase(self, direction):
        slice = self.pizza.slice_at(self.cursor_position)
        new_slice = self.pizza.increase(slice, direction, self.max_ingredients_per_slice)
        if (new_slice is not None and min(self.pizza.ingredients.of(new_slice)) >=
            self.min_each_ingredient_per_slice):

            if slice in self.valid_slices:
                self.valid_slices.remove(slice)
            self.valid_slices.append(new_slice)
            score = self.score_of(new_slice) - self.score_of(slice)
            self.score += score
            return score * self.rewards['POSITIVE']['MADE_SQUARE']
        return self.rewards['NEUTRAL']['SUB_OPTIMAL_SLICE'] if new_slice is not None else self.rewards['NEGATIVE']['BACKTRACK']


    def move(self, direction):
        self.next_cursor_position = tuple(np.add(self.cursor_position, self.delta_position[direction]))
        
        reward_quality = 0
        # Profiles: Slice Mode/ No Slice Mode
        if self.slice_mode:
            reward_quality += self.increase(direction)
        elif(self.movement_in_bounds()):
            reward_quality += self.moved_out_of_slice()
            reward_quality += self.did_backtrack()
            reward_quality += self.movement()
        else:
            reward_quality += self.rewards['NEGATIVE']['MOVED_OUT_OF_BOUNDS']
        return reward_quality

    def did_double_toggle(self):
        if self.prev_action == 'toggle':
            self.double_toggle_count += 1
            return self.rewards['NEGATIVE']['DOUBLE_TOGGLE']
        else:
            return 0
    
    def toggled_after_slice_exit(self):
        x, y = self.previous_position
        slice_map = self.pizza._map
        if (slice_map[x][y] != -1):
            return self.rewards['POSITIVE']['TOGGLED_ON_FIRST_ACTION_OUT_OF_SLICE']
        else:
            return self.rewards['NEUTRAL']['MOVED']
        
    def toggle(self, action):
        reward_quality = 0

        reward_quality += self.did_double_toggle()
        reward_quality += self.toggled_after_slice_exit()

        self.slice_mode = not self.slice_mode
        return reward_quality

    def do(self, action):
        self.next_action = action

        direction = Direction[action] if action != 'toggle' else 'toggle'
        reward = getattr(self, self.reward_context[action])(direction)
        


        #Update actions
        self.prev_action = self.curr_action
        self.curr_action = self.next_action
        self.next_action = 'none'

        #Update Game Step
        self.current_game_step += 1
        return reward

    def state(self):
        return {
            'ingredients_map': self.pizza.ingredients._map.tolist(),
            'slices_map': self.pizza._map.tolist(),
            'cursor_position': self.cursor_position,
            'slice_mode': self.slice_mode,
            'min_each_ingredient_per_slice': self.min_each_ingredient_per_slice,
            'max_ingredients_per_slice': self.max_ingredients_per_slice,
        }

def gen_random_board(rows, cols):
    selection = ['M', 'T']
    return [''.join(np.random.choice(selection) for i in range(cols)) for j in range(rows)]


#ROWS = 12
#COLS = 12
#
#board = gen_random_board(ROWS, COLS)
#
#
#pizza_config = {
#    'pizza_lines': board,
#    'r': ROWS,
#    'c': COLS,
#    'l': 1,
#    'h': 6
#}
#
#g = GoogleEngineer(pizza_config)


'''


init_config = {
    'pizza_lines': board,
    'r': ROWS,
    'c': COLS,
    'l': 1,
    'h': 6
}

g = GoogleEngineer(init_config)
g.do('up')
g.do('toggle')
g.move((0, 1))
'''