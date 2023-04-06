import random 
import numpy as np
import math
import matplotlib.pyplot as plt


class model():
    '''
    A model class
    Contains the users and all information related
    '''
    def __init__(self,x,y,capacity=100):
        '''
        Initializes the model object 
        x (int) :size of model
        y (int) :size of model
        capacity (int) : maximum numbers of user allowed in model
        '''
        self.x = x
        self.y = y
        self.users = []
        self.capacity = capacity
        
    def get_capacity(self):
        '''
        Method to get capacity
        '''
        return self.capacity
    
    def add_user(self,user):
        '''
        Add a user to the users list
        '''
        self.users.append(user)
        
    def plot(self):
        '''
        plot the model with users at random locations
        
        '''
        x_coords = [np.random.uniform(self.x-1, self.x+1) for user in self.users]
        y_coords = [np.random.uniform(self.y-1, self.y+1) for user in self.users]
        colors = np.random.rand(len(self.users))
        plt.scatter(x_coords, y_coords, c=colors)
        plt.show()

