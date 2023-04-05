from BP_reward import get_reward
import numpy as np
np.random.seed(0)

#Doctor A:
def doctor_a():
    rewards = []
    actions = []
    id= []
#randomly selecting first action

    action = np.random.randint(low=1,high=3)

    p_sw=0.8
    p_sl=0.7

    for i in range(1,101):

        reward = get_reward(action,810100352)
        actions.append(action)
        id.append(i)
        rewards.append(reward)

        if reward > 0:
            action = np.random.choice([action,(3-action)],p=[p_sw,1-p_sw])
        elif reward < 0:
            action = np.random.choice([action,(3-action)],p=[1-p_sl,p_sl])
        else:
            action = np.random.randint(low=1,high=3)
#averaging rewards
    cum_reward = np.cumsum(rewards)
    A_rewards = np.stack([id,actions,cum_reward],axis=1)

    return A_rewards

# #Doctor B:
def doctor_b():
    rewards = []
    actions = []
    id= []

    for i in range(1,101):
        action = np.random.randint(low=1,high=3)
        reward = get_reward(action,810100352)
        rewards.append(reward)
        actions.append(action)
        id.append(i)

    cum_reward = np.cumsum(rewards)
    B_rewards = np.stack([id,actions,cum_reward],axis=1)
    
    return B_rewards

#Doctor C:
def doctor_c():
    rewards = []
    actions = []
    id= []
#repeating first action for 10 times
    action=1
    for i in range(1,11):
        reward = get_reward(action,810100352)
        rewards.append(reward)
        actions.append(action)
        id.append(i)
#repeating second action for 10 times
    action=2
    for i in range(11,21):
        reward = get_reward(action,810100352)
        rewards.append(reward)
        actions.append(action)
        id.append(i)
#finding the action with maximum reward
    idmax=np.argmax(rewards)
    action = actions[idmax]

    i+=1
#repeating the loop until i==100
    while i < 101:
#repeating the action with maximum reward for 7 times
        for j in range(1,8):

            reward = get_reward(action,810100352)
            rewards.append(reward)
            actions.append(action)
            id.append(i)
            i+=1
#randomly selecting action for 3 times
        for j in range(1,4):

            action = np.random.randint(low=1,high=3)
            reward = get_reward(action,810100352)
            rewards.append(reward)
            actions.append(action)
            id.append(i)
            i+=1
#finding the action with maximum reward
        idmax = np.argmax(rewards)
        action = actions[idmax]

    cum_reward = np.cumsum(rewards)
    C_rewards = np.stack([id,actions,cum_reward],axis=1)

    return C_rewards