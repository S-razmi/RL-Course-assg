import numpy as np

def get_reward(action , student_id ):
    alpha1 = int(str(student_id)[6:])%3 + 1
    beta1 = 5

    alpha2= 2
    beta2 = alpha1

    if action==1:

        stochastic_num1 = np.random.beta(alpha1, beta1 , size=None)
        reward = -stochastic_num1 *14 + 7
        return reward

    elif action ==2:
    
        stochastic_num1 = np.random.beta(alpha2, beta2 , size=None)
        reward = stochastic_num1 *14 - 7
        return reward

    else: 
        print('Action number must be 1 or 2!')
    
    


