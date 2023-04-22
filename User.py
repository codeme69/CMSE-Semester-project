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
        relation credibility index (float): an overall index of how credible the user realtions are
        misinformation_rate (float): score/number of relations
        '''
        self.score = score
        self.gender = gender
        self.age = age
        self.relations = [] #list of all the users that this user is related to
        self.carrier = False
        self.receiver = receiver
        self.relation_credibility_index = 0
        self.misinformation_rate = 0
    
    def get_name(self):
        '''
        Method to get the name of user
        '''
        return self.name 
    
    def set_name(self,name):
        '''
        Method to get the name of user
        '''
        self.name = name
    
    def get_age(self):
        '''
        Method to get the age of user
        '''
        return self.age
    def set_age(self,age):
        '''
        Method to get the age of user
        '''
        self.age=age
    
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
        self.score = np.random.random()
        
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
    
    def set_color(self,color="lightblue"):
        self.color = color
        
    def get_color(self):
        return self.color
    
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
    
    def get_relation_credit(self):
        return self.relation_credibility_index
    
    def set_relation_credit(self, info):
        self.relation_credibility_index = info
        
    def set_misinformation_rate(self,rate):
        '''
        Method to set the rate of user
        '''
        self.misinformation_rate = rate
    
    def get_misinformation_rate(self):
        '''
        Method to get the reate of user
        '''
        return self.misinformation_rate
    
    def set_next_state(self,value):
        '''
        sets the predicted value if user is probable to share true or false information next time
        '''
        self.next_state = value
        
    def get_next_state(self):
        '''
        gets the predicted value if user is probable to share true or false information next time
        '''
        return self.next_state
        
    def get_properties(self):
        
        #bool
        previous_info_shared = random.randint(0, 1)
        return [self.score,len(self.relations),self.relation_credibility_index,self.misinformation_rate,previous_info_shared]


