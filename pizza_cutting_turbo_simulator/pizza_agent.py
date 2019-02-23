from src.game import Game
import random


def reset():
    default_args = {'max_steps': 100}
    pizza_lines = ['TTTTT', 'TMMMT', 'TTTTT']
    pizza_config = { 'pizza_lines': pizza_lines, 'r': 3 , 'c': 5, 'l': 1, 'h': 6 }
    
    game = Game(default_args)
    game.init(pizza_config)
    
    return game


actionspace = ['up', 'down', 'left', 'right', 'toggle']
game = reset()

game.render()
while not game.env['done']:
    action = random.choice(actionspace)
    print(action, type(action))
    game.step(action)
    game.render()


print(game.max_steps)





