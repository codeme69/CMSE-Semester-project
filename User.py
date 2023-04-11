import random 
import numpy as np
import math
import matplotlib.pyplot as plt
import time  
from IPython.display import display, clear_output
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import string


class User():
    '''
    An User class
    
    '''
    def __init__(self,gender='M',age=18,score=0.6,receiver=False):
        '''
        Initializes the user class
        name(str) : name of user
        gender(char) : gender of user
        age (int) : age of the user
        score(float) :  fraction of how many times user has shared accurate information on social media
        carrier (bool) : true if carrying information
        reciever (bool) : true if this user needs that information
        '''
        self.score = score
        self.gender = gender
        self.age = age
        self.relations = [] #list of all the users that this user is related to
        self.carrier = False
        self.receiver = receiver
        
        self.name = self.generate_random_short_name(5)
    
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
        
    def is_carrier(self):
        return self.carrier
    
    def set_carrier(self,is_carrier):
        self.carrier = is_carrier
        
    def is_receiver(self):
        return self.receiver
    
    def set_position(self,pos):
        self.position=pos
        
    def get_position(self):
        return self.position
    
    def generate_random_short_name(self,length=5):
        """
        Generates a random short name of given length.
        Args:
            length (int): Length of the random short name. Default is 3.
        Returns:
            str: Randomly generated short name.
        """
        # Generate random lowercase letters
        letters = string.ascii_lowercase
        random_name = ''.join(random.choice(letters) for _ in range(length))
        return random_name
