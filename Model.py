import random 
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import string


# +
#pip install decorator==5.0.9 
# -

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
        pos = (random.uniform(0,self.x), random.uniform(0,self.y))
        user.set_position(pos)
        self.users.append(user)
        
    def set_relations(self):
        
        # Get the list of users from the model
        users =self.users

        # Add random relations between users
        for user in users:
            # Select a random number of neighbors for each user
            num_relations = random.randint(1, len(self.users))
            
            if (user.get_color()=="blue") or (user.get_color()=="red") :
                num_relations = 2
                
            # Select random users from the list as relations
            relations = random.sample(users, num_relations)

            # Add the selected neighbors as relations
            for relation in relations:
                # Skip adding self as neighbor
                if (relation == user):
                    continue
                
               
                user.add_relation(relation)
                
    def check_consecutive_pair(self,shortest_path, pair):
        """
        Check if a pair of values are in a shortest_path as consecutive elements
        pair (tuple): Tuple of two values to check as consecutive elements in the list (order doesn't matter).

        Returns:
            bool: True if the pair of values are found as consecutive elements in the list, False otherwise.
        """
        if len(shortest_path) < 2:
            return False

        pair1 = pair
        pair2 = pair[::-1]  # Reverse the pair
        
        for i in range(len(shortest_path) - 1):
            if (shortest_path[i].get_name() == pair1[0].get_name() and shortest_path[i + 1].get_name() == pair1[1].get_name()) or (shortest_path[i].get_name() == pair2[0].get_name() and shortest_path[i + 1].get_name() == pair2[1].get_name()):

                return True

        return False

    
    def plot(self,shortest_path=None):
        '''
        Plot the model with users and relationships
        '''


        plt.figure(figsize=(20, 20))  # Set the size of the plot
        
        # Create a networkx graph to represent the model
        G = nx.Graph()
        
        non_participants = []
        mediate_users = []
        node_colors = []
        edge_colors = []
        
         # Add nodes to the graph
        for user in self.users:
            G.add_node(user)  
            node_colors.append(user.get_color())
            if user not in shortest_path:
                non_participants.append(user)
            else:
                mediate_users.append(user)
                
        
        # Generate random x and y coordinates for each node
        pos = nx.random_layout(G)
        
        #add edges that are part of shortest path
        for i in range(len(shortest_path)-1):
            user = shortest_path[i]
            next_user = shortest_path[i+1]
            misinformation_weight = user.get_score()+next_user.get_score()
            edge_colors.append("green")
            G.add_edge(user, next_user,weight = 3,color="green")
        
        # Add relationships as edges that are not part of shortest path to the graph
        for user in self.users:
                for relation in user.get_relations():
                    if not G.has_edge(user,relation) and not G.has_edge(relation,user):
                        misinformation_weight = user.get_score()+relation.get_score()
                        edge_colors.append("gray")
                        G.add_edge(user, relation,weight = 0.5,color="gray")

            
        edges = G.edges()
        colors = [G[u][v]['color'] for u,v in edges]
        weights = [G[u][v]['weight'] for u,v in edges]
        
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos,nodelist=self.users, node_size=300, node_color=node_colors, edgecolors='black', linewidths=1)
        
        # Draw the edges
        nx.draw_networkx_edges(G, pos,edge_color=colors, width=weights)
        
        # Add labels to the nodes
        labels = {user: user.get_name() for user in self.users}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black')
        
        # Add a legend
        info_patch = mpatches.Patch(color='lightblue', label='User')
        path_patch = mpatches.Patch(color='green', label='Shortest Path')
        receiver_patch = mpatches.Patch(color='red', label='Receiver node')
        sender_patch = mpatches.Patch(color='blue', label='Sender node')
        plt.legend(handles=[info_patch, path_patch,receiver_patch,sender_patch])
        
        
        plt.axis('off')  # Turn off the axis
        plt.show()  # Show the plot
        
    def reset_plot(self, event):
        '''
        Reset the plot by clearing the current figure and resetting the shortest path
        '''
        plt.clf()
        self.plot()


