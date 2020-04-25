import numpy as np
from collections import Counter
import random



def get_the_state(state_hunter, state_scout, x_hunter, x_scout, y_hunter, y_scout, radius, radius_scout):
    test = False
    for i in state_hunter:
        if 1 in i:
            test = True

    if test:
        #print("State hunter : "+str(state_hunter))
        state_hunter
    
    mid = int(len(state_scout)/2)
    x_rel = x_hunter - (x_scout - radius)
    y_rel = y_hunter - (y_scout - radius)
    
    x_prey = state_scout[0]
    y_prey = state_scout[1]

    '''
    print("X_hunter  : "+str(x_hunter))
    print("Y_hunter  : "+str(y_hunter))
    print("X_scout  : "+str(x_scout))
    print("Y_scout  : "+str(y_scout))
    '''
    x_prey_abs = (x_scout+x_prey)%10
    y_prey_abs = (y_scout+y_prey)%10
    
    dx = 0
    dy = 0
    
    if abs((x_hunter-x_scout))<abs((10-x_hunter+x_scout)):
        dx = x_scout - x_hunter
    else:
        dx = 10-x_hunter+x_scout
    if abs((y_hunter-y_scout))<abs((10-y_hunter+y_scout)):
        dy = y_scout - y_hunter
    else:
        dy = 10-y_hunter+y_scout
    
    
    if abs((x_hunter-x_prey_abs))<abs((10-x_hunter+x_prey_abs)):
        dx_final = x_prey_abs - x_hunter
    else:
        dx_final = 10-x_hunter+x_prey_abs
    if abs((y_hunter-y_prey_abs))<abs((10-y_hunter+y_prey_abs)):
        dy_final = y_prey_abs - y_hunter
    else:
        dy_final = 10-y_hunter+y_prey_abs
    '''
    print("X HUNTER -> SCOUT = "+str(dx))
    print("Y HUNTER -> SCOUT = "+str(dy))
    print("X SCOUT -> PREY = "+str(x_prey))
    print("Y SCOUT -> PREY = "+str(y_prey))
    print("X HUNTER -> PREY = "+str(dx_final))
    print("Y HUNTER -> PREY =  "+str(dy_final))
    '''
    return [dx_final, dy_final]
    
def distance(x1, x2, y1, y2, size):
    if abs((x1-x2))<abs((size-x1+x2)):
            dx = x2 - x1
    else:
        dx = size-x1+x2
    if abs((y1-y2))<abs((size-y1+y2)):
        dy = y2 - y1
    else:
        dy = size-y1+y2
    return [dx, dy]
    

def dist_from_center(state, radius):
     
    r = np.where(np.array(state) > 0)
    '''
    print("state = ")
    for i in state:
        print(i)
    print("r = " +str(r))
    '''
    try:
        x_prey = r[0] - radius
        y_prey = radius - r[1]
        #print(" distance : "+str([x_prey, y_prey]))
        return [x_prey, y_prey]
    except:
        return 0

def mean_tables(tables):
    all_keys = []
    results  = {}
    for i in tables:
        all_keys += i.keys()
    all_keys = list(set(all_keys))
    for i in all_keys:
        results[i] = {}
        for j in ["up", "down", "left", "right", "stay"]:
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
    def __init__(self, posx, posy, state_size, action_size, beta, gamma, typer, grid_width, grid_length ,intelligent=False, world_wraps=False, epsilon=0, decay_rate=0, save=False):
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
            self.beta = beta
            self.gamma = gamma
            self.rewards = []
            self.actions = []
            self.states = []
            self.epsilon = epsilon
            self.decay_rate = decay_rate
            
            self.actions_history = []
            self.rewards_history = []
            self.states_history = []
            self.save = save
    def pprint(self):
        if self.type=="dead": return ""
        ret = ""
        ret += "Type = "+str(self.type)+" / Intelligent = "+str(self.intelligent)+"\n"
        ret += "( X="+str(self.posx)+" / Y="+str(self.posy)+" )"+"\n"
        ret += str(self.steps)+" steps."
        if self.intelligent:
            ret += "Beta = "+str(self.beta)+" / Gamma  = "+str(self.gamma)+" / Decay Rate = "+str(self.decay_rate)+"\n"
            ret += "History size = "+str(len(self.actions_history))+"\n"
        ret += "\n"
        return ret

    def choose(self, state=None):
        if self.type=="dead": return
        self.steps += 1
        if not self.intelligent:
            direction = random.choice(["up", "down", "left", "right", "stay"])
            return direction
        if random.uniform(0, 1) < self.epsilon:
            direction = random.choice(["up", "down", "left", "right", "stay"])
        else:
            if str(state) in self.Q:
                direction = max(self.Q[str(state)], key=self.Q[str(state)].get) 
            else:
                self.Q[str(state)] = {}
                self.Q[str(state)]['up'] = 0
                self.Q[str(state)]['down'] = 0
                self.Q[str(state)]['left'] = 0
                self.Q[str(state)]['right'] = 0
                self.Q[str(state)]['stay'] = 0
                direction = max(self.Q[str(state)], key=self.Q[str(state)].get) 
        if self.intelligent:  
            self.actions.append(direction)
            self.states.append(state)

        return direction
    
    def place(self, x, y):
        if self.type=="dead": return
        self.posx = x
        self.posy = y
        if self.intelligent:
            self.epsilon -= self.decay_rate
        self.steps = 0
        
    def optimal_value(self, state):
        try:
            maximum = max(self.Q[state], key=self.Q[state].get)
            return (self.Q[state][maximum])
        except:
            return 0

    
    def update_q_table(self, reward, action, state, new_state, Q=None):
        if self.type=="dead": return
        if Q!=None:
            self.Q = Q
        if str(state) in self.Q:
            self.Q[str(state)][action] = self.Q[state][action] + self.beta * (reward + self.gamma * self.optimal_value(new_state) - self.Q[str(state)][action])
        else:
            self.Q[str(state)] = {}
            self.Q[str(state)]['up'] = 0
            self.Q[str(state)]['down'] = 0
            self.Q[str(state)]['left'] = 0
            self.Q[str(state)]['right'] = 0
            self.Q[str(state)]['stay'] = 0
            self.Q[str(state)][action] = self.Q[state][action] + self.beta * (reward + self.gamma * self.optimal_value(new_state) - self.Q[str(state)][action])
        
        if self.save:
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
            self.update_q_table()
    
    def get_memory(self):
        if self.type=="dead": return
        return [self.states_history, self.rewards_history, self.actions_history]
        
    def move(self, direction):
        if self.type=="dead": return
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
            elif direction == "down": #and self.posy>0:
                self.posy -= 1
                
            elif direction == "left": #and self.posx>0:
                self.posx -= 1
                
            elif direction == "right": #and self.posx<self.grid_width:
                self.posx += 1
                
            else:
                pass        
            if self.posy>=self.grid_length:
                self.posy = 0
            if self.posy<0:
                self.posy = self.grid_length-1
            if self.posx<0:
                self.posx = self.grid_width-1
            if self.posx>=self.grid_width:
                self.posx = 0
            
class RL:
    def __init__(self, beta, gamma, grid_width, grid_length, radius=4, radius_scout=2, world_wraps = False, sharing_q_table=False, mean_frequency=0, number_to_catch=1, epsilon=0, decay_rate=0, communicating_hunters=False):
        self.world_wraps = world_wraps
        self.radius = radius
        self.radius_scout = radius_scout
        self.communicating_hunters = communicating_hunters
        self.agents = []
        self.state_size = 50
        self.action_size = 4
        self.beta = beta
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
        self.grid = []
        self.mean = 0
        self.mean50 = 0
        self.end = False

    def get_grid(self):
        grid = np.zeros((self.grid_width, self.grid_length), dtype=np.uint64)
        #print("shape  : "+str(grid.shape))
        for i in self.agents:
            if i.type == "prey":
                #print("posx :"+str(i.posx)+" posy :"+str(i.posy)) 
                grid[i.posy, i.posx] = 1
        return grid
    
    def get_state(self, posx, posy, hunter=False):
        self.get_grid()
        state = []
        if not hunter:
            rad = self.radius_scout
        else:
            rad = self.radius
        for x in range(posx-rad, posx+rad+1):
            ranger = []
            for y in range(posy-rad, posy+rad+1):
                if not self.world_wraps:
                    if x>=0 and y>=0 and x<self.grid_width and y<self.grid_length:
                        ranger.append(int(self.grid[y, x]))
                else:
                    ranger.append(int(self.grid[y%(self.grid_length), x%(self.grid_width) ]  ))
            state.append(ranger)
            state = dist_from_center(state, self.radius)
        if hunter and len(state) != 0 and self.scouts>0:
            print("Scouty")
            for j in self.agents:
                if j.type == "scout": # IF THERE IS A SCOUT ADD HIS PERCEPTION TO THE STATE
                    state = get_the_state(state, self.get_state(j.posx, j.posy, hunter=False), posx, posx, posy, j.posy, self.radius, self.radius_scout)
        return str(state)

    def iteration(self):
        if not self.communicating_hunters:
            for i in self.agents:   
                if i.intelligent:
                    state = self.get_state(i.posx, i.posy, hunter=True)
                    action = i.choose(state)
                    i.move(action)
                    new_state = self.get_state(i.posx, i.posy, hunter=True)
                    if self.is_end_episode():
                        reward = 1
                    else:
                        reward = -0.1
                    i.update_q_table(reward ,action, state, new_state)
                else:
                    i.move(i.choose())    
        else:
            states = {}
            distances = {}
            for i in self.agents:
                if i.intelligent:
                      states[str(self.agents.index(i))] = dist_from_center(self.get_state(i.posx, i.posy))
                      distances[str(self.agents.index(i))] = []
                      for j in self.agents:
                          if j.intelligent and self.agents.index(j)!=self.agents.index(i):
                              distances[str(self.agents.index(i))].append(distance(i.posx, j.posx, i.posy, j.posy,  self.grid_length)) 
            for i in self.agents:
                if i.intelligent:
                    choices.append(i.choose(str([list(states.values()), distances[str(self.agents.index(i))] ])))
                else:
                    i.move(i.choose())
               
        if self.is_end_episode():
            self.episode_number += 1

            self.reinit()
            r = self.steps 
            if self.mean_frequency>0: # IF THERE IS A FREQUENCY OF SYNCHRONIZATION
                if self.episode_number%self.mean_frequency == 0:
                    qss = [] 
                    result  = mean_tables(qss)
                    for i in self.agents:
                        i.Q = result
            self.mean = ((self.mean*(self.episode_number-1))+r)/self.episode_number
            if self.episode_number%50==0:
                self.mean50 = 0
            else:
                self.mean50 = ((self.mean50*(self.episode_number%50-1))+r)/(self.episode_number%50)
            
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
            self.end = True
            return True
        self.end = False
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
        agent.update_q_table()

    def reinit(self):
        for i in self.agents:
            i.place(np.random.randint(1, 10), np.random.randint(1, 10))

    def pprint(self):    
        ret = ""    
        ret += "Episode "+str(self.episode_number)+"\n"
        ret += "Sharing-Q-Table = "+str(self.sharing_q_table)+"\n"
        ret += "Mean-Frequency =  "+str(self.mean_frequency)+"\n"
        ret += "Number to catch = "+str(self.number_to_catch)+"\n"

        ret += "Mean overall = " +str(self.mean)+"\n"
        ret += "Mean on last 50 episodes  = "+str(self.mean50)+"\n\n"

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
        
        
    def print_q(self, ida):
        lis = list(self.agents[ida].Q.keys())
        print("\n\nKeys")
        for i in lis:
            print(str(i))
            
    def add_hunter(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.beta, self.gamma, "hunter", self.grid_width, self.grid_length, intelligent=True, world_wraps=self.world_wraps, epsilon=self.epsilon, decay_rate=self.decay_rate)
        self.agents.append(ag)

    def add_scout(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.beta, self.gamma, "scout", self.grid_width, self.grid_length, intelligent=False, world_wraps=self.world_wraps)
        self.agents.append(ag)
        self.scouts += 1
        
    def add_expert_hunter(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.beta, self.gamma, "expert", self.grid_width, self.grid_length, intelligent=True, world_wraps=self.world_wraps, epsilon=self.epsilon, decay_rate=self.decay_rate)
        self.agents.append(ag)
    
    def add_prey(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.beta, self.gamma, "prey", self.grid_width, self.grid_length, world_wraps=self.world_wraps)
        self.agents.append(ag)
        