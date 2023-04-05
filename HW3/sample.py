from env import FrozenLake
import numpy as np

# Action Space 
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTION_SPACE = [LEFT,DOWN,RIGHT,UP]

Discount = 0
Sd_number = 810100352

environment = FrozenLake(studentNum=Sd_number)

# optional 
environment.render()

# value iteration
for state in allstates:
    for action in ACTION_SPACE:
        for state_p in Nextstates:
            action_values[action] +=  probability * (reward + Discount * V[state_p])
            ...




