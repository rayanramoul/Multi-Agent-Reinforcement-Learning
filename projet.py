import numpy as np
import random
def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y
class Agent:
    def __init__(self, posx, posy, state_size, action_size, learning_rate, gamma, typer, grid_width, grid_length ,intelligent=False, world_wraps=False):
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
            self.epsilon = -1
            self.rewards = []
            self.actions = []
            self.states = []
            
        
    def choose(self, state=None):
        # Set the percent you want to explore
        #if state is not None:
        #    #state = int("".join(str(x) for x in state), 2)
        #    state = bool2int(state)
        if state is  not None:
            state = np.array(state)
        #    print(state)
            state = str(np.argwhere(state>0))
        #    print(state)
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
        #print("Q-Table : \n"+str(self.Q))
        self.posx = x
        self.posy = y
        if self.intelligent:
            self.epsilon -= 0.0025
        self.steps = 0
        
    def optimal_future_value(self, i):
        try:
            state = self.states[i]
            maximum = max(self.Q[state], key=self.Q[state].get)  # Just use 'min' instead of 'max' for minimum.
            return (self.Q[state][maximum])
        except:
            return 0
    
    def update(self):
        #print("states len "+str(self.states))
        #print("rewards len "+str(self.rewards))
        #print("actions len "+str(self.actions))
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
        #print("Q-Table\n"+str(self.Q))
        self.rewards = []
        self.actions = []
        self.states = []
        
    def move(self, direction):
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
    def __init__(self, learning_rate, gamma, grid_width, grid_length, world_wraps = False):
        self.world_wraps = world_wraps
        self.agents = []
        self.state_size = 50
        self.action_size = 4
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.episode_number = 1
        self.steps = 0
    def add_hunter(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "hunter", self.grid_width, self.grid_length, intelligent=True, world_wraps=self.world_wraps)
        self.agents.append(ag)

    def get_grid(self):
        grid = np.zeros((self.grid_width+1, self.grid_length+1), dtype=np.uint64)
        for i in self.agents:
            if i.type == "prey":
                grid[i.posx, i.posy] = 1
        return grid
    
    def get_state(self, posx, posy, radius=4):
        #print(" For position ("+str(posx)+", "+str(posy)+")")
        grid = self.get_grid()
        state = []
        for x in range(posx-radius, posx+radius+1):
            for y in range(posy-radius, posy+radius+1):
                if not self.world_wraps:
                    if x>=0 and y>=0 and x<self.grid_width+1 and y<self.grid_length+1:
                        #print("("+str(x)+", "+str(y)+") : "+str(grid[x, y]))
                        state.append(grid[x, y])
                else:
                    #print("("+str(x%(self.grid_width+1))+", "+str(y%(self.grid_length+1))+") : "+str(grid[x%(self.grid_width+1), y%(self.grid_length+1)]))
                    state.append(grid[x%(self.grid_width+1), y%(self.grid_length+1)])
                
        #print("State : "+str(state))
        return state
    
    def add_prey(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "prey", self.grid_width, self.grid_length, world_wraps=self.world_wraps)
        self.agents.append(ag)
    
    def episode(self):
        choices = []
        for i in self.agents:
            if i.type == "hunter":
                state = self.get_state(i.posx, i.posy) 
                choices.append(i.choose(state))
            else:
                choices.append(i.choose())
        for i in self.agents:
            i.move(choices[self.agents.index(i)])
        for i in self.agents:
            if i.type == "hunter":
                self.reward(i)
        if self.is_end_episode():
            self.episode_number += 1
            #print("Episode "+str(self.episode_number))
            self.reinit()
            r = self.steps 
            #print(" Steps : "+str(r))
            self.steps = 0
            return r
        self.steps += 1
        return 0
        
    
    def is_end_episode(self):
        preys_coord = []
        for i in self.agents:
            if i.type == "prey":
                preys_coord.append((i.posx, i.posy))

        for i in self.agents:
            if i.type == "hunter" and (i.posx, i.posy) in preys_coord:
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
        #print("Reinit ! ")
        for i in self.agents:
            if i.type == "hunter":
                i.update()
            i.place(np.random.randint(1, 10), np.random.randint(1, 10))

    def pprint(self):
        print(str(self.grid.representation))

