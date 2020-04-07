import numpy as np
import random

class Agent:
    def __init__(self, posx, posy, state_size, action_size, learning_rate, gamma, typer, grid_width, grid_length ,intelligent=False):
        self.Q = np.zeros((state_size, action_size))
        self.posx = posx
        self.posy = posy
        self.lr = learning_rate
        self.gamma = gamma
        self.type = typer
        self.grid_width = grid_width
        self.grid_length = grid_length
    def choose(self):
        direction = random.choice(["up", "down", "left", "right", "nothing"])
        self.move(direction)
    
    def place(self, x, y):
        self.posx = x
        self.posy = y
        
    def update(self, state, action, new_state, reward):
        self.Q[state, action] = Q[state, action] + self.lr * (reward + self.gamma * np.max(Q[new_state, :]) - Q[state, action])

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
        
    def add_hunter(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "hunter", self.grid_width, self.grid_length, intelligent=True)
        self.agents.append(ag)

    def get_grid(self):
        grid = np.zeros((self.grid_width, self.grid_length))
        for i in self.agents:
            if i.type == "hunter":
                grid[i.posx, i.posy] = 2
            else:
                grid[i.posx, i.posy] = 1
        return grid
    
    def get_state(self, posx, posy, radius=2):
        grid = self.get_grid()
        state = [posx, posy]
        for x in range(posx-radius, posx+radius):
            for y in range(posy-radius, posy+radius):
                state.append(grid[x, y])
                
        print("State : "+str(state))
        return state
    
    def add_prey(self, posx, posy):
        ag = Agent(posx, posy, self.state_size, self.action_size, self.learning_rate, self.gamma, "prey", self.grid_width, self.grid_length)
        self.agents.append(ag)
    
    def episode(self):
        for i in self.agents: 
            i.choose()
        if self.is_end_episode():
            self.reinit()
    
    def is_end_episode(self):
        preys_coord = []
        for i in self.agents:
            if i.type == "prey":
                #print(" i :"+str(self.agents.index(i))+" "+str(i.type)+" coord :"+str(i.posx)+" / "+str(i.posy))
                state = self.get_state(i.posx, i.posy)
                preys_coord.append((i.posx, i.posy))
        #print("Preys coord : "+str(preys_coord))
        for i in self.agents:
            if i.type == "hunter" and (i.posx, i.posy) in preys_coord:
                return True
        return False
    
    def reinit(self):
        print("Reinit ! ")
        for i in self.agents:
            i.place(np.random.randint(1, 10), np.random.randint(1, 10))
    def pprint(self):
        print(str(self.grid.representation))

