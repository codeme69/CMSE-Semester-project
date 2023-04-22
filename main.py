from User import User 
from Model import model
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


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
        G.add_node(user)
        for relation in user.get_relations():
            misinformation_weight = user.get_score()+relation.get_score()
            G.add_edge(user, relation,weight = misinformation_weight)
           
    if source is None or target is None:
        print("Error: At least one user with information and one user that needs information should be present")
        return None
    
    if (nx.has_path(G,source,target)):
        # Find the shortest path between the user with information and the user that needs information
        shortest_path = nx.shortest_path(G, source, target)
    
        return shortest_path
    
    return None



# -

df = pd.read_csv('Data.csv')
df_short = df.head(50)
df_short

# +
mod = model(200, 200)

# Add users to the model
user1 = User()
user1.set_carrier(True)
user1.set_color("blue")
user1.set_name("Sender")
mod.add_user(user1)

for i in range(50):
    user = User()
    user.set_color()
    user.set_name("{}".format(i+3))
    mod.add_user(user)
    
user2 = User(receiver=True)
user2.set_color("red")
user2.set_name("Receiver")
mod.add_user(user2)
mod.set_relations()

# Call best_path and plot functions
shortest_path = best_path(mod.users,user1,user2)
for user in shortest_path:
    print(user.get_name())
if shortest_path:
    mod.plot(shortest_path)

# -

