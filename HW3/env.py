import numpy as np
from gym import Env
import matplotlib.pyplot as plt


def make_map(studentNum,size=(6,6),mode=1):
    assert mode in [1,2],"Mode should be either 1 for deterministic 2 for random"
    """
    Observation Space : map of environment 
    A 6*6 Grid Graph
    Start point = (0,0)
    End point = (5,5)
    Just one safe path for each Student number
    The probability of falling in each state:
    in safe path = 0.0001 and for other states = 1

    :param 1: Student number 
    
    :return : The Created Map 
    """

    np.random.seed(studentNum)  

    move = np.zeros(size[0]+size[1]-2)  # Minimum moves for start to the end point based on the size of the graph
    idx = np.random.choice(range(size[0]+size[1]-2),size=size[1]-1,replace=False)
    move[idx] = 1

    point = [0,0]
    lowprobs = [tuple(point)]

    for m in move:
        if m:
            point[0] += 1
        else:
            point[1] += 1
        lowprobs.append(tuple(point))

    idx = np.array(lowprobs)
    #based on the input mode breaking probablity of unsafe cells are either random or 1
    #if mode is 1 breaking probability of safe cells are 0.0001 and if it's 2 probability is 0.001
    if mode==1:
        map = np.ones(size)
        map[idx[:,0],idx[:,1]] = 0.0001 
        map[0,0] = 0.0   # Start point
        map[size[0]-1,size[1]-1] = 0.0 
    elif mode==2:
        map = np.random.random(size)
        map[idx[:,0],idx[:,1]] = 0.001
        map[0,0] = 0.0   # Start point
        map[size[0]-1,size[1]-1] = 0.0  
    return map

class FrozenLake(Env):
    def __init__(self,studentNum,gamma,theta,slip_prob,size=(6,6),mode=1):
        #GET THE MAP OF THE LAKE
        self.obs = make_map(studentNum=studentNum,size=size,mode=mode)
        #CALCULATE THE NUMBER OF STATES BASED ON THE MAP (WE HAVE ONE ADDITIONAL STATE REPRESENTING TERMINAL STATE )
        self.n_states=self.obs.flatten().shape[0]+1
        self.states = [i for i in range(self.n_states)]
        #S VARIABLE KEEPS THE STATE WE ARE IN (IT'S JUST FOR THE STEP FUNCTION AND DOESN'T HAVE ANY OTHER FUNCTIONALITY)
        self.s=0
        #DISCOUNT FACTOR
        self.discount_factor = gamma
        #PROBABLITIY OF WHETHER THE AGENT WILL SLIP ON ICE OR NOT
        self.slip_prob = slip_prob
        self.ACTION= {'UP':3,'DOWN':1,'RIGHT':2,'LEFT':0}
        #THETA VALUE  FOR POLICY AND VALUE ITERATION
        self.theta = theta
        n_actions = len(self.ACTION)
        #POLICY VARIABLE CONTAINS THE CURRENT POLICY AND IT'S INITIATED WITH ALL ZEROS
        self.policy = np.zeros(shape=(self.n_states,n_actions))
        #V VARIABLE CONTAINTS THE CURRENT VALUES OF THE STATES BASED ON THE MOST RECENT POLICY AND IT'S INTITATED WITH ALL ZEROS
        self.V = np.zeros(self.n_states)
        #Q VARIABLE CONTAINS STATE ACTION VALUES FOR ALL STATES AND ACTIONS
        self.Q = np.zeros(shape=(self.n_states,n_actions))
        #T VARIABLE STORES THE TRANSITION PROBABILITY BETWEEN ALL STATES
        self.T = np.zeros((self.n_states, n_actions,self.n_states )) 
        #R VARIABLE STORES THE REWARD FOR EACH ACTION IN BETWEEN EACH TWO STATES
        self.R = np.zeros((self.n_states, n_actions,self.n_states ))
        n_rows = self.obs.shape[0]
        n_cols = self.obs.shape[1]
        #WE CALCULATE THE TRANSITION PROBABILIIES FOR EACH STATE AND STORE IT IN IT'S ELEMENT IN T VARIABLE
        #LOOP OVER ALL CELLS IN THE MAP
        for r in range(n_rows):
            for c in range(n_cols):
                """IF COLUMN INDEX OF THE CELL IS ZERO (aka cell is next to the left fence) 
                FOR ACTION LEFT AGENT GETS BACK TO THE SAME CELL WITH PROB =1
                OTHERWISE AGENT WILL GO TO EITHER IT'S LEFT CELL OR TO TERMINAL STATE BASED ON THE LEFT CELL'S BREAKING PROBABILITY
                """
                if c==0:
                    self.T[r*n_rows+c,self.ACTION['LEFT'],r*n_rows+c] = 1.0
                else:
                    self.T[r*n_rows+c,self.ACTION['LEFT'],r*n_rows+c-1]= 1 - self.obs[r][c-1]
                    self.T[r*n_rows+c,self.ACTION['LEFT'],self.states[-1]] = self.obs[r][c-1]
                #FOR CELLS NEXT TO RIGHT FENCE AGENT GOES BACK TO IT'S CELL WITH PROB 1
                if c==n_cols-1:
                    self.T[r*n_rows+c,self.ACTION['RIGHT'],r*n_rows+c] = 1.0
                else:
                    #OTHER WISE EITHER TO IT'S RIGHT CELL OR TO TERMINAL 
                    self.T[r*n_rows+c,self.ACTION['RIGHT'],r*n_rows+c+1]= 1 - self.obs[r][c+1]
                    self.T[r*n_rows+c,self.ACTION['RIGHT'],self.states[-1]] = self.obs[r][c+1]
                #CELLS BELOW THE TOP FENCE
                if r==0:
                    self.T[r*n_rows+c,self.ACTION['UP'],r*n_rows+c] = 1.0
                else:
                    self.T[(r)*n_rows+c,self.ACTION['UP'],(r-1)*n_rows+c]= 1 - self.obs[r-1][c]
                    self.T[r*n_rows+c,self.ACTION['UP'],self.states[-1]] = self.obs[r-1][c]
                #CELLS ABOVE THE BOTTOM FENCE
                if r==n_rows-1:
                    self.T[r*n_rows+c,self.ACTION['DOWN'],r*n_rows+c] = 1.0
                else:
                    self.T[(r)*n_rows+c,self.ACTION['DOWN'],(r+1)*n_rows+c]= 1 - self.obs[r+1][c]
                    self.T[r*n_rows+c,self.ACTION['DOWN'],self.states[-1]] = self.obs[r+1][c]
        #WHEN AGENT REACHES GOAL OR TERMINAL STATE IT CAN'T GET OUT WITH ANY ACTION SO WITH PROBABILITY OF 1 GOES BACK IT IT'S CELL
        self.T[self.n_states-2,:,:] = 0
        self.T[self.n_states-2,:,self.n_states-2] = 1
        self.T[self.n_states-1,:,self.n_states-1] = 1

        for r in range(n_rows):
            for c in range(n_cols):

                if c==0:
                    #WHEN AGENT HIT THE FENCE IT GETS BACK TO IT CELL WITH REWARD ZERO
                    self.R[r*n_rows+c,self.ACTION['LEFT'],r*n_rows+c] = 0
                else:
                    #WHEN AGENT GOES TO OTHER NON GOAL CELLS IT GET EITHER A -1 REWARD IF THEY DON'T BREAK OR -11 IF THEY BREAK
                        self.R[r*n_rows+c,self.ACTION['LEFT'],self.states[-1]] = -11
                        self.R[r*n_rows+c,self.ACTION['LEFT'],r*n_rows+c-1]= -1

                if c==n_cols-1:
                    self.R[r*n_rows+c,self.ACTION['RIGHT'],r*n_rows+c] = 0
                else:
                        self.R[r*n_rows+c,self.ACTION['RIGHT'],self.states[-1]] = -1
                        self.R[r*n_rows+c,self.ACTION['RIGHT'],r*n_rows+c+1]= -1
                    
                if r==0:
                    self.R[r*n_rows+c,self.ACTION['UP'],r*n_rows+c] = 0
                else:
                        self.R[r*n_rows+c,self.ACTION['UP'],self.states[-1]] = -11
                        self.R[(r)*n_rows+c,self.ACTION['UP'],(r-1)*n_rows+c] = -1
            
                if r==n_rows-1:
                    self.R[r*n_rows+c,self.ACTION['DOWN'],r*n_rows+c] = 0
                else:
                        self.R[r*n_rows+c,self.ACTION['DOWN'],self.states[-1]] = -11
                        self.R[(r)*n_rows+c,self.ACTION['DOWN'],(r+1)*n_rows+c]= -1
        #REWARD FOR CELL ABOVE AND LEFT OF GOAL STATE IS 99 FOR ACTIONS GETTING THEM TO GOAL STATE
        self.R[self.n_states - n_rows - 2,self.ACTION['DOWN'],self.n_states-2]= 99
        self.R[self.n_states-3,self.ACTION['RIGHT'],self.n_states-2]= 99
        #ACTIONS IN TERMINAL AND GOAL STATES DON'T HAVE ANY REWARD
        self.R[self.n_states-2,:] = 0
        self.R[self.n_states-1,:] = 0

    def reset(self):
        self.policy = np.zeros(shape=(self.n_states,len(self.ACTION)))
        self.V = np.zeros(self.n_states)
        self.Q = np.zeros(shape=(self.n_states,len(self.ACTION)))

    def step(self,input_action):
        #THIS FUNCTION IS JUST FOR DEMONSTRATION PURPOSES AND DOESN'T HAVE ANY ROLE IN SOLVING THE GIVEN PROBLEM
        assert input_action in self.ACTION, "Input Action must be LEFT,RIGHT,UP,DOWN"
        #GET THE ACTION
        action = self.ACTION[input_action]
        #SAVE THE LAST STATE
        old_state = self.s
        #WITH A GIVEN PROBABILITY SLIP OR DON'T SLIP
        if np.random.random() > self.slip_prob:
            #DON'T SLIP
            transition_prob = self.T[self.s,action,:]
            self.s = np.random.choice(self.states,p=transition_prob)
        else:
            #SLIP
            #CHOOSE A RANDOM ACTION
            action = np.random.choice(len(self.ACTION))
            #GET THE PROBABILITY
            transition_prob = self.T[self.s,action,:]
            #GO TO DESTINATION CELL  BASED ON THE TRANSITION PROBABILITY
            self.s = np.random.choice(self.states,p=transition_prob)
        #GET THE REWARD
        reward = self.R[old_state,action,self.s]
        #IF THE DESTINATION CELL IS TERMINAL (ICE BROKE) GO BACK TO INITIAL STATE
        terminate = False
        if self.s == self.states[-1]:
            terminate = True
            self.s = 0
        #RETURN THE STATE, REWARD AND IF IT TERMINATED
        return self.s,reward,terminate
 
    def render(self):
        nrows, ncols = self.obs.shape
        values = self.V[:-1].reshape(nrows,ncols)
        plt.figure(figsize=(10,10))
        idx=0
        plt.matshow(self.obs,fignum=1,cmap='Pastel2')
        actions = {0:[0,1],1:[0,-1],2:[1,0],3:[-1,0]}
        for i in range(nrows):
            for j in range(ncols):
                val = round(values[i,j],2)
                arrow= actions[np.argmax(self.policy[idx])]
                break_prob = np.round(self.obs[i,j],4)
                
                # print(idx,val,np.argmax(policy[idx]),arrow)
                
                plt.text(y=i-0.3,x=j+0.2,s=f'Value {val}',va='center',ha='center',)
                plt.text(y=i+0.2,x=j+0.2,s=break_prob,va='center',ha='center',)
                plt.text(y=i+0.4,x=j+0.2,s=f'Num: {idx+1}',va='center',ha='center',)
                plt.quiver(j,i,arrow[0],arrow[1],scale=25)
                idx += 1
        plt.show()

    # def close(self): 
    #     """
    #     (Optional) : Perform cleanup

    #     """
    