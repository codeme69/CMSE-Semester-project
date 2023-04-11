from User import User 
from Model import model
import networkx as nx
import matplotlib.pyplot as plt


# +
#function for best path to transmit information

def best_path(users,source,target):
    '''
    Find the shortest path between users with information and users that need that information
    
    users (list): List of User objects
    
    Returns:
    - shortest_path (list): List of User objects representing the shortest path from the user with information to the user that needs that information
    '''
    # Create a directed graph to represent the relationships between users
    G = nx.DiGraph()
    for user in users:
        for relation in user.get_relations():
            G.add_edge(user, relation)
               
    if source is None or target is None:
        print("Error: At least one user with information and one user that needs information should be present")
        return None
    
    # Find the shortest path between the user with information and the user that needs information
    shortest_path = nx.shortest_path(G, source, target)
    
    return shortest_path



# +
mod = model(50, 50)

# Add users to the model
user1 = User()
user1.set_carrier(True)
mod.add_user(user1)

user2 = User(receiver=True)
mod.add_user(user2)

for i in range(10):
    user = User()
    mod.add_user(user)
    
mod.set_relations()

# Call best_path and plot functions
shortest_path = best_path(mod.users,user1,user2)
mod.plot(shortest_path)

# -


