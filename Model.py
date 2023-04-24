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
import time
from IPython.display import display, clear_output


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
        G (network graph) : graph that holds and plots user nodes
        node colors (list) : list of colors for user nodes
        statements (list of strings): print the misformation weights info
        '''
        self.x = x
        self.y = y
        self.users = []
        self.capacity = capacity
        self.G = nx.Graph()
        self.node_colors = []
        self.statements = []
        
    def get_capacity(self):
        '''
        Method to get capacity
        '''
        return self.capacity
    
    def get_users(self):
        '''
        Method to get capacity
        '''
        return self.users
    
    def add_user(self,user):
        '''
        Add a user to the users list
        '''
        self.users.append(user)
        self.G.add_node(user)  
        self.node_colors.append(user.get_color())
        
    def set_relations(self):
        
        # Get the list of users from the model
        users =self.users

        # Add random relations between users
        for user in users:
            # Select a random number of neighbors for each user
            num_relations = random.randint(1, round(len(self.users)*0.5))
            
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
                
    def set_user_properties(self):
         # Get the list of users from the model
        users =self.users

        # Add random relations between users
        for user in users:
            
            pos = (random.uniform(0,self.x), random.uniform(0,self.y))
            score = random.random()
            age = random.uniform(18,35)
            relation_creditbility_index = np.mean([i.get_score() for i in user.get_relations()])
            misinformation_rate = score*relation_creditbility_index*len(user.get_relations())
            
            user.set_position(pos)
            user.set_score(score)
            user.set_age(age)
            user.set_relation_credit(relation_creditbility_index)
            user.set_misinformation_rate(misinformation_rate)
            
        
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

    def print_statements(self):
        '''
        prints statements of misinforamtion weight
        
        supposed to be print in plot function but interrupts plotting of graph
        '''
        for i in self.statements:
            print(i)

    def plot(self, shortest_paths):
        '''
        Plot the model with users and relationships
        '''
        
        node_colors = []
        edge_colors = []


        # Generate random x and y coordinates for each node
        pos = nx.random_layout(self.G)

        # Add relationships as edges that are not part of shortest path to the graph
        for user in self.users:
            for relation in user.get_relations():
                if not self.G.has_edge(user, relation) and not self.G.has_edge(relation, user):
                    misinformation_weight = user.get_score() + relation.get_score()
                    edge_colors.append("gray")
                    self.G.add_edge(user, relation, weight=misinformation_weight, color="gray")

        # Add labels to the nodes
        labels = {user: user.get_name() for user in self.users}
        
        plt.figure(figsize=(15, 15))    
        shortest_path_edgelist = []
        color_list = []  #list of colors in order for each short path edge
        shortest_path_colors = ["green","blue","black","red","pink","orange"]

        # Add green edges one by one and pause for 1 second between each drawing
        for j,shortest_path in enumerate(shortest_paths):
            for i in range(len(shortest_path) - 1):
                
                plt.figure(figsize=(15, 15))   
                user = shortest_path[i]
                next_user = shortest_path[i + 1]
                shortest_path_edgelist.append((user, next_user))
                color_list.append(shortest_path_colors[j])
                
                misinformation_weight = user.get_score() + next_user.get_score()
                
                edge_colors.append(shortest_path_colors[j])
                self.G.add_edge(user, next_user, weight=misinformation_weight, color=shortest_path_colors[j])
                
                edges = self.G.edges()
                colors = [self.G[u][v]['color'] for u, v in edges]
                weights = [self.G[u][v]['weight'] for u, v in shortest_path_edgelist]
                
                # Draw the nodes
                nx.draw_networkx_nodes(self.G, pos, nodelist=self.users, node_size=300, node_color= self.node_colors, edgecolors='black',
                                       linewidths=1)
                # Draw the edges
                nx.draw_networkx_edges(self.G, pos , edge_color=edge_colors, width=0.5)
                nx.draw_networkx_labels(self.G, pos, labels, font_size=12, font_color='black')
                nx.draw_networkx_edges(self.G, pos, edgelist= shortest_path_edgelist, edge_color=color_list, width=3)
                
                for edge, weight in zip(shortest_path_edgelist, weights):
                    if edge in shortest_path_edgelist:
                        edge_pos = pos[edge[0]] + (pos[edge[1]] - pos[edge[0]]) / 2
                        plt.text(edge_pos[0], edge_pos[1], f"MIW: {weight:.2f}", color='black', fontsize=12,
                                ha='center', va='center')
                
                # Add a legend
                info_patch = mpatches.Patch(color='lightblue', label='User')
                path_patch = mpatches.Patch(color='green', label='Shortest Path')
                receiver_patch = mpatches.Patch(color='red', label='Receiver node')
                sender_patch = mpatches.Patch(color='blue', label='Sender node')
                miw_patch = mpatches.Patch(color="black",label='MIW : Misiformation weight')
                plt.legend(handles=[info_patch, path_patch, receiver_patch, sender_patch,miw_patch])

                plt.pause(0.01)
                clear_output(wait=True)
        
                plt.show()
        index = 0
        weights = [self.G[u][v]['weight'] for u, v in shortest_path_edgelist]
        for i,shortest_path in enumerate(shortest_paths):
            length = len(shortest_path)-1
            total_weight = round(sum(weights[index:length+index]),2)
            index = length
            self.statements.append("The total misinformation weight in "+str(shortest_path_colors[i])+" path is "+ str(total_weight))
