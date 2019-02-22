from gym.envs.registration import register

register(
    id='pizza-v0',
    entry_point='gym_pizza.envs:PizzaEnv'
)
register(
    id='pizza-extrahard-v0',
    entry_point='gym_pizza.envs:PizzaExtraHardEnv'
)