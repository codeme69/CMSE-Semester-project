import random 
import numpy as np
import math
import matplotlib.pyplot as plt
import time  
from IPython.display import display, clear_output
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data


# +
class User():
    '''
    An User class
    
    '''
    def __init__(self,name,gender='M',age=18,score=0.6):
        '''
        Initializes the user class
        name(str) : name of user
        gender(char) : gender of user
        age (int) : age of the user
        score(float) : score that represents fraction of how many times user has shared accurate information on social media
        
        '''
        self.name = name
        self.score = score
        self.gender = gender
        self.age = age
        self.relations = [] #list of all the users that this user is related to
    
    def get_name(self):
        '''
        Method to get the name of user
        '''
        return self.name
    
    def get_age(self):
        '''
        Method to get the age of user
        '''
        return self.age
    
    def get_score(self):
        '''
        Method to get the score of user
        '''
        return self.score
    
    def set_score(self,score):
        '''
        Method to set the score of user
        score(float) : updated user score 
        '''
        self.score = score
        
    def add_relation(self,user):
        '''
        Method to add a new related user
        '''
        self.relations.append(user)
    
    def get_relations(self):
        '''
        Method to get a list of relations of the user
        '''
        return self.relations
        
    
