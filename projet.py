import numpy as np
from collections import Counter
import random

def transform(state):
    state = np.array(state)
    #print("im here")
    #print(state)
    x = np.argwhere(state>0)
    try:
        return str(x[0][0])
    except:
        return str(0)
    
    
def mean_tables(tables):
    all_keys = []
    results  = {}
    for i in tables:
        all_keys += i.keys()
    all_keys = list(set(all_keys))
    print("all keys : "+str(all_keys))
    for i in all_keys:
        results[i] = {}
        for j in ["up", "down", "left", "right"]:
            summ = 0
            nbr = 0
            for k in tables:
                if i in k.keys() and k[i][j]!=0:
                    summ += k[i][j]
                    nbr += 1
            if nbr != 0:
                summ /= nbr
            results[i][j] = summ
            
    return results
    
class Agent:
    def __init__(self, posx, posy, state_size, action_size, learning_rate, gamma, typer, grid_width, grid_length ,intelligent=False, world_wraps=False, epsilon=0, decay_rate=0):
        self.Q = {}
        self.world_wraps = world_wraps
        self.posx = posx
        self.posy = posy
        self.type = typer
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.intelligent = intelligent
        self.steps = 0
        if self.intelligent:
            self.lr = learning_rate
            self.gamma = gamma
            self.rewards = []
            self.actions = []
            self.states = []
            self.epsilon = epsilon
            self.decay_rate = decay_rate
            
            self.actions_history = []
            self.rewards_history = []
            self.states_history = []

    def pprint(self):
        if self.type=="dead": return ""
        ret = ""
        ret += "Type = "+str(self.type)+" / Intelligent = "+str(self.intelligent)+"\n"
        ret += "( X="+str(self.posx)+" / Y="+str(self.posy)+" )"+"\n"
        ret += str(self.steps)+" steps."
        if self.intelligent:
            ret += "Learning Rate = "+str(self.lr)+" / Gamma  = "+str(self.gamma)+" / Decay Rate = "+str(self.decay_rate)+"\n"
            ret += "History size = "+str(len(self.actions_history))+"\n"
        ret += "\n"
        
        return ret

    def choose(self, state=None):
        if self.type=="dead": return

        #print("state received : "+str(state))
        self.steps += 1
        if not self.intelligent:
            direction = random.choice(["up", "down", "left", "right", "nothing"])
            #self.move(direction)
            return direction
        if random.uniform(0, 1) < self.epsilon:
            direction = random.choice(["up", "down", "left", "right"])
            #self.move(direction)
        else:
            if str(state) in self.Q:
                direction = max(self.Q[str(state)], key=self.Q[str(state)].get) 
                #print(str(self.Q[str(state)]))
                #self.move(direction)
            else:
                self.Q[str(state)] = {}
                self.Q[str(state)]['up'] = 0
                self.Q[str(state)]['down'] = 0
                self.Q[str(state)]['left'] = 0
                self.Q[str(state)]['right'] = 0
                direction = max(self.Q[str(state)], key=self.Q[str(state)].get) 
                #self.move(direction)
        if self.intelligent:  
            #print("Epsilon : "+str(self.epsilon))      
            self.actions.append(direction)
            self.states.append(state)
        #print("Direction :"+str(direction))
        return direction
    
    def place(self, x, y):
        if self.type=="dead": return

        #print("Q-Table : \n"+str(self.Q))
        self.posx = x
        self.posy = y
        if self.intelligent:
            self.epsilon -= self.decay_rate
        self.steps = 0
        
    def optimal_future_value(self, i):
        try:
            state = self.states[i]
            maximum = max(self.Q[state], key=self.Q[state].get)  # Just use 'min' instead of 'max' for minimum.
            return (self.Q[state][maximum])
        except:
            return 0
    
    def update(self, Q=None):
        if self.type=="dead": return

        #print("states len "+str(self.states))
        #print("rewards len "+str(self.rewards))
        #print("actions len "+str(self.actions))
        if Q!=None:
            self.Q = Q
        for i in range(len(self.rewards)):
            #print("iteration : "+str(i))
            if str(self.states[i]) in self.Q:
                self.Q[str(self.states[i])][self.actions[i]] = self.Q[str(self.states[i])][self.actions[i]] + self.lr * (self.rewards[i] + self.gamma * self.optimal_future_value(i+1) - self.Q[str(self.states[i])][self.actions[i]])
            else:
                self.Q[str(self.states[i])] = {}
                self.Q[str(self.states[i])]['up'] = 0
                self.Q[str(self.states[i])]['down'] = 0
                self.Q[str(self.states[i])]['left'] = 0
                self.Q[str(self.states[i])]['right'] = 0
                self.Q[str(self.states[i])]['nothing'] = 0
                self.Q[str(self.states[i])][self.actions[i]] = self.Q[str(self.states[i])][self.actions[i]] + self.lr * (self.rewards[i] + self.gamma * self.optimal_future_value(i+1) - self.Q[str(self.states[i])][self.actions[i]])
        self.rewards_history.append(self.rewards)
        self.states_history.append(self.states)
        self.actions_history.append(self.actions)
        
        self.rewards = []
        self.actions = []
        self.states = []
        
    def replay_memory(self, states, rewards, actions):
        if self.type=="dead": return

        for i in range(len(states)):
            self.actions = actions[i]
            self.states = states[i]
            self.rewards = rewards[i]
            self.update()
    
    def get_memory(self):
        if self.type=="dead": return
        return [self.states_history, self.rewards_history, self.actions_history]
        
    def move(self, direction):
        if self.type=="dead": return
        #print("Q-Table\n"+str(self.Q))
        if not self.world_wraps:
            if direction == "up" and self.posy<self.grid_length:
                self.posy += 1
            elif direction == "down" and self.posy>0:
                self.posy -= 1
            elif direction == "left" and self.posx>0:
                self.posx -= 1
            elif direction == "right" and self.posx<self.grid_width:
                self.posx += 1
            else:
                pass
        else:
            if direction == "up": #and self.posy<self.grid_length:
                self.posy += 1
                if self.posy>self.grid_length:
                    self.posy = 0
            elif direction == "down": #and self.posy>0:
                self.posy -= 1
                if self.posy<0:
                    self.posy = self.grid_length
            elif direction == "left": #and self.posx>0:
                self.posx -= 1
                if self.posx<0:
                    self.posx = self.grid_width
            elif direction == "right": #and self.posx<self.grid_width:
                self.posx += 1
                if self.posx>self.grid_width:
                    self.posx = 0
            else:
                pass
class RL:
    def __init__(self, learning_rate, gamma, grid_width, grid_length, radius=4, world_wraps = False, sharing_q_table=False, mean_frequency=0, number_to_catch=1, epsilon=0, decay_rate=0):
        self.world_wraps = world_wraps
        self.radius = radius
        self.agents = []
        self.state_size = 50
        self.action_size = 4
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.episode_number = 1
        self.steps = 0
        self.scouts = 0
        self.sharing_q_table = sharing_q_table
        self.Q = {}
        self.mean_frequency = mean_frequency
        self.number_to_catch = number_to_catch
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        
    def add_hunter(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "hunter", self.grid_width, self.grid_length, intelligent=True, world_wraps=self.world_wraps, epsilon=self.epsilon, decay_rate=self.decay_rate)
        self.agents.append(ag)

    def add_scout(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "scout", self.grid_width, self.grid_length, intelligent=False, world_wraps=self.world_wraps)
        self.agents.append(ag)
        self.scouts += 1
        
    def add_prey(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "prey", self.grid_width, self.grid_length, world_wraps=self.world_wraps)
        self.agents.append(ag)
        
    def get_grid(self):
        grid = np.zeros((self.grid_width+1, self.grid_length+1), dtype=np.uint64)
        for i in self.agents:
            if i.type == "prey":
                grid[i.posx, i.posy] = 1
        return grid
    
    def get_state(self, posx, posy):
        #print(" For position ("+str(posx)+", "+str(posy)+")")
        grid = self.get_grid()
        state = []
        for x in range(posx-self.radius, posx+self.radius+1):
            for y in range(posy-self.radius, posy+self.radius+1):
                if not self.world_wraps:
                    if x>=0 and y>=0 and x<self.grid_width+1 and y<self.grid_length+1:
                        #print("("+str(x)+", "+str(y)+") : "+str(grid[x, y]))
                        state.append(grid[x, y])
                else:
                    #print("("+str(x%(self.grid_width+1))+", "+str(y%(self.grid_length+1))+") : "+str(grid[x%(self.grid_width+1), y%(self.grid_length+1)]))
                    state.append(grid[x%(self.grid_width+1), y%(self.grid_length+1)])
                
        #print("State : "+str(state))
        return state

    def episode(self):
        choices = []
        for i in self.agents:
            if i.intelligent:
                state = transform(self.get_state(i.posx, i.posy)) 
                if self.scouts>0:
                    for j in self.agents:
                        if j.type == "scout": # IF THERE IS A SCOUT ADD HIS PERCEPTION TO THE STATE
                            state = [state, transform(self.get_state(j.posx, j.posy)), i.posx-j.posx, i.posy-j.posy] 
                choices.append(i.choose(state))
            else:
                choices.append(i.choose())
        for i in self.agents:
            i.move(choices[self.agents.index(i)])
        for i in self.agents:
            if i.intelligent:
                self.reward(i)
        if self.is_end_episode():
            self.episode_number += 1
            print("Episode "+str(self.episode_number))
            self.reinit()
            r = self.steps 
            #print(" Steps : "+str(r))
            if self.mean_frequency>0: # IF THERE IS A FREQUENCY OF SYNCHRONIZATION
                if self.episode_number%self.mean_frequency == 0:
                    qss = []        #print("Reinit ! ")
                    result  = mean_tables(qss)
                    for i in self.agents:
                        i.Q = result
                
            self.steps = 0
            return r
        self.steps += 1
        return 0
        
    
    def is_end_episode(self):
        preys_coord = []
        for i in self.agents:
            if i.type == "prey":
                preys_coord.append((i.posx, i.posy))
        count = 0
        for i in self.agents:
            if i.type == "hunter" and (i.posx, i.posy) in preys_coord:
                count += 1
        
        if count>=self.number_to_catch:
            return True
        return False
    
    def reward(self, agent):
        preys_coord = []
        rew = -0.1
        for i in self.agents:
            if i.type == "prey":
                preys_coord.append((i.posx, i.posy))
        if (agent.posx, agent.posy) in preys_coord:
                rew = 1
        agent.rewards.append(rew)
    
    def reinit(self):
        for i in self.agents:
            if i.intelligent:
                if not self.sharing_q_table:
                    i.update()
                else: # IF THERE IS SHARED Q-TABLE
                    i.update(self.Q)
            i.place(np.random.randint(1, 10), np.random.randint(1, 10))

    def pprint(self):    
        ret = ""    
        ret += "Episode "+str(self.episode_number)+"\n"
        ret += "Sharing-Q-Table = "+str(self.sharing_q_table)+"\n"
        ret += "Mean-Frequency =  "+str(self.mean_frequency)+"\n"
        ret += "Number to catch = "+str(self.number_to_catch)+"\n\n"

        for i in self.agents:
            if i.type!="dead":
                ret += "Agent "+str(self.agents.index(i))+"\n"
                ret += i.pprint()
                ret += "\n"
        return ret
    
    def delete_agent(self, index):
        self.agents[index].type = "dead"
    
    def teach(self, teacher, student):
        mem = self.agents[int(teacher)].get_memory()
        self.agents[int(student)].replay_memory(mem[0], mem[1], mem[2])