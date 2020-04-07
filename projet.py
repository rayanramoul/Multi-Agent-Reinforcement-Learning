import numpy as np
import random

class Agent:
    def __init__(self, posx, posy, state_size, action_size, learning_rate, gamma, typer, grid_width, grid_length ,intelligent=False):
        self.Q = {}
        self.posx = posx
        self.posy = posy
        self.lr = learning_rate
        self.gamma = gamma
        self.type = typer
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.epsilon = 1
        
        self.rewards = []
        self.actions = []
        self.states = []
        self.steps = 0
        
    def choose(self, state):
        # Set the percent you want to explore
        self.steps += 1
        if random.uniform(0, 1) < self.epsilon:
            direction = random.choice(["up", "down", "left", "right", "nothing"])
            self.move(direction)
        else:
            try:
                direction = max(self.Q[str(state)], key=self.Q[str(state)].get) 
            except KeyError:
                self.Q[str(state)] = {}
                self.Q[str(state)]['up'] = 0
                self.Q[str(state)]['down'] = 0
                self.Q[str(state)]['left'] = 0
                self.Q[str(state)]['right'] = 0
                self.Q[str(state)]['nothing'] = 0
                direction = max(self.Q[str(state)], key=self.Q[str(state)].get) 
                
        self.actions.append(direction)
        self.states.append(state)
        
    def place(self, x, y):
        #print("Q-Table : \n"+str(self.Q))
        self.posx = x
        self.posy = y
        self.epsilon -= 0.001
        self.steps = 0
        
    def optimal_future_value(self):
        maxes = []
        for i in self.Q:
            maximum = max(self.Q[i], key=self.Q[i].get)  # Just use 'min' instead of 'max' for minimum.
            maxes.append(self.Q[i][maximum])
        return max(maxes)
    
    def update(self):
        #print("states len "+str(len(self.states)))
        #print("rewards len "+str(len(self.rewards)))
        #print("actions len "+str(len(self.actions)))
        for i in range(len(self.rewards)):
            try:
                self.Q[str(self.states[i])][self.actions[i]] = self.Q[str(self.states[i])][self.actions[i]] + self.lr * (self.rewards[i] + self.gamma * self.optimal_future_value() - self.Q[str(self.states[i])][self.actions[i]])
            except KeyError:
                self.Q[str(self.states[i])] = {}
                self.Q[str(self.states[i])]['up'] = 0
                self.Q[str(self.states[i])]['down'] = 0
                self.Q[str(self.states[i])]['left'] = 0
                self.Q[str(self.states[i])]['right'] = 0
                self.Q[str(self.states[i])]['nothing'] = 0
                self.Q[str(self.states[i])][self.actions[i]] = self.Q[str(self.states[i])][self.actions[i]] + self.lr * (self.rewards[i] + self.gamma * self.optimal_future_value() - self.Q[str(self.states[i])][self.actions[i]])

        self.rewards = []
        self.actions = []
        self.states = []
        
    def move(self, direction):
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
class RL:
    def __init__(self, learning_rate, gamma, grid_width, grid_length):
        self.agents = []
        self.state_size = 50
        self.action_size = 4
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.episode_number = 1
    def add_hunter(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "hunter", self.grid_width, self.grid_length, intelligent=True)
        self.agents.append(ag)

    def get_grid(self):
        grid = np.zeros((self.grid_width+1, self.grid_length+1))
        for i in self.agents:
            if i.type == "prey":
                grid[i.posx, i.posy] = 1
        return grid
    
    def get_state(self, posx, posy, radius=4):
        grid = self.get_grid()
        state = []
        for x in range(posx-radius, posx+radius):
            for y in range(posy-radius, posy+radius):
                if x>0 and y>0 and x<self.grid_width and y<self.grid_length:
                    state.append(grid[x, y])
                
        #print("State : "+str(state))
        return state
    
    def add_prey(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "prey", self.grid_width, self.grid_length)
        self.agents.append(ag)
    
    def episode(self):
        for i in self.agents:
            state = self.get_state(i.posx, i.posy) 
            i.choose(state)
        for i in self.agents:
            self.reward(i)
        if self.is_end_episode():
            self.episode_number += 1
            print("Episode "+str(self.episode_number))
            self.reinit()
        
    
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
        print("Reinit ! ")
        for i in self.agents:
            i.update()
            i.place(np.random.randint(1, 10), np.random.randint(1, 10))

    def pprint(self):
        print(str(self.grid.representation))

