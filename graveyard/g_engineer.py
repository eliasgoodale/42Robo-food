from enum import Enum
import numpy as np 

class Direction(Enum):
    up = 0
    down = 1
    left = 2
    right = 3

    @classmethod
    def opposite(cls, direction):
        return ({
            cls.right: cls.left,
            cls.down: cls.up,
            cls.left: cls.right,
            cls.up: cls.down,
        })[direction]

class GoogleEngineer:
    delta_position = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
    }

    reward_context = {
            'up': 'move',
            'down': 'move',
            'left': 'move',
            'right': 'move',
            'toggle': 'modal'
    }

    rewards = {
        'POSITIVE': 1.0,
        'NEGATIVE': -1.0,
        'NEUTRAL': 0,
    }
    def __init__(self, pizza_config):
        #self.pizza = Pizza(pizza_config['pizza_lines'])

        self.r=pizza_config['r']
        self.c=pizza_config['c']
        self.min_ingred = pizza_config['l']
        self.max_ingred = pizza_config['h']
        
        # Game memory
        self.prev_action = 'none'
        self.current_game_step = 0
        
        self.previous_position = (-1, -1)
        self.cursor_position = (0, 0)
        self.next_cursor_position = (-1, -1)

        # Punishment action counts
        self.shuffle_count = 0
        self.double_toggle_count = 0

        # Positional Data

        self.slice_mode = False
        self.valid_slices = []
        self.score = 0
    
    '''
        For directional actions we want to make sure we are passing negative
        rewards for being inside of the slice map 
    '''
    def movement_in_bounds(self, x, y):
        return (0 <= y <= self.r) and (0 <= x <= self.c)


    def move(self, direction):
        next_cursor_position = tuple(np.add(self.cursor_position, self.delta_position[direction]))

        if (self.movement_in_bounds(next_cursor_position[0], next_cursor_position[1])):

            print ('Action Valid: ', next_cursor_position)
        else:
            print ('Action Invalid: ', next_cursor_position)

    
    def modal(self, action):

        


    def do(self, action):
        reward = getattr(self, self.reward_context[action])(action)



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

g = GoogleEngineer(init_config)
g.do('up')
g.do('toggle')
#g.move((0, 1))