import random 
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.patches as mpatches
import string


# +
#pip install decorator==5.0.9 

# +
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

            # Select random users from the list as relations
            relations = random.sample(users, num_relations)

            # Add the selected neighbors as relations
            for relation in relations:
                # Skip adding self as neighbor
                if relation == user:
                    continue

                # Add the neighbor as a relation
                user.add_relation(relation)
                
    def check_consecutive_pair(self,pair):
        """
        Check if a pair of values are in a users list as consecutive elements
        pair (tuple): Tuple of two values to check as consecutive elements in the list (order doesn't matter).

        Returns:
            bool: True if the pair of values are found as consecutive elements in the list, False otherwise.
        """
        if len(self.users) < 2:
            return False

        pair1 = pair
        pair2 = pair[::-1]  # Reverse the pair

        for i in range(len(self.users) - 1):
            if (self.users[i] == pair1[0] and self.users[i + 1] == pair1[1]) or (self.users[i] == pair2[0] and self.users[i + 1] == pair2[1]):
                    
                return True

        return False


    def plot(self,shortest_path=None):
        '''
        Plot the model with users and relationships
        '''
        plt.figure(figsize=(10, 10))  # Set the size of the plot
        
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
        
        # Add relationships as edges to the graph
        for user in self.users:
            for relation in user.relations:
                G.add_edge(user.name, relation.name)
                if self.check_consecutive_pair((user,relation)):
                    edge_colors.append("green")
                else:
                    edge_colors.append("gray")
        
        # Generate random x and y coordinates for each node
        pos = nx.random_layout(G)
        
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos,nodelist=self.users, node_size=300, node_color=node_colors, edgecolors='black', linewidths=1)
        
        # Draw the edges
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=1)
        
        # Add labels to the nodes
        labels = {user: user.get_name() for user in self.users}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black')
        
#         # Draw the shortest path
#         if shortest_path:
#             shortest_path_nodes = [user.name for user in shortest_path]
#             colors = [user.get_color() for user in shortest_path]
#             nx.draw_networkx_nodes(G, pos, nodelist=shortest_path_nodes, node_size=300, node_color=colors, edgecolors='black', linewidths=1)
#             nx.draw_networkx_edges(G, pos, edgelist=[(shortest_path[i-1].name, shortest_path[i].name) for i in range(1, len(shortest_path))], edge_color='green', width=3)
        
        # Add a legend
        info_patch = mpatches.Patch(color='lightblue', label='User')
        path_patch = mpatches.Patch(color='green', label='Shortest Path')
        receiver_patch = mpatches.Patch(color='red', label='Receiver node')
        sender_patch = mpatches.Patch(color='blue', label='Sender node')
        plt.legend(handles=[info_patch, path_patch,receiver_patch,sender_patch])
        
        # Add a reset button
#         reset_ax = plt.axes([0.92, 0.01, 0.07, 0.05])
#         reset_button = widgets.Button(reset_ax, 'Reset', color='lightgray')
#         reset_button.on_clicked(self.reset_plot)
        
        plt.axis('off')  # Turn off the axis
        plt.show()  # Show the plot
        
    def reset_plot(self, event):
        '''
        Reset the plot by clearing the current figure and resetting the shortest path
        '''
        plt.clf()
        self.plot()



