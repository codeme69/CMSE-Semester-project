from User import User 
from Model import model
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn import svm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.decomposition import PCA


# ## Best Path Algorithm

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
        shortest_path = nx.shortest_path(G, source, target,weight="weight")
    
        return shortest_path
    
    return None



# -

# ## Creating and Populating the model with users

# +
mod = model(200, 200)

sender_list = []
# Add senders to the model
for i in range(5):
    user = User()
    user.set_carrier(True)
    user.set_color("blue")
    user.set_name("Sender")
    mod.add_user(user)
    sender_list.append(user)

#populate the model
for i in range(100):
    user = User()
    user.set_color()
    user.set_name("{}".format(i+3))
    mod.add_user(user)
    
#add the receiver to model
receiver_user = User(receiver=True)
receiver_user.set_color("red")
receiver_user.set_name("Receiver")
mod.add_user(receiver_user)

mod.set_relations()
mod.set_user_properties()
# -
# ## Model Animation

# +
# Call best_path and plot functions
shortest_paths = []

#find shortest paths from all senders to receiver
for sender in sender_list:
    shortest_path = best_path(mod.users,sender,receiver_user)
    shortest_paths.append(shortest_path)

#plot the model
mod.plot(shortest_paths)

# -


# #### Misinformation weight of the shortest paths

mod.print_statements()

# # Building Machine Learning Model

# ### Loading the manual dataset

df = pd.read_csv("cmse202_dataset.csv")
df

# ### Creating the training and testing data for ML model

label = df["next_shared_info_prediction"]
Features = df.drop(["next_shared_info_prediction", "gender","age","user"], axis=1)
train_labels, test_labels, train_vectors, test_vectors = train_test_split(label, Features, test_size=0.25, train_size=0.75)

# ### Running Logistic regression and SVM Model

# +
logit_model = sm.Logit(train_labels, sm.add_constant(train_vectors))
result = logit_model.fit()

# Summarize the model
print(result.summary())
# -

# #### Following cell takes some time (3-4mins) as the data set produced is mostly random-based...the model is not quite accurate yet

# +
#make some temporary variables so you can change this easily
tmp_vectors = train_vectors
tmp_labels = train_labels

print("Fitting the classifier to the training set")
# a dictionary of hyperparameters: key is the name of the parameter, value is a list of values to test
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.01, 0.1,1.0,10.0],
              'kernel': ['linear','rbf']}
# make a classifier by searching over a classifier and the parameter grid
clf = GridSearchCV(SVC(class_weight='balanced'), param_grid,n_jobs=-1)

# we have a "good" classifier (according to GridSearchCV), how's it look
clf = clf.fit(tmp_vectors, tmp_labels)
print("Best estimator found by grid search:")
print(clf.best_estimator_)
print("Best parameters found by grid search:")
print(clf.best_params_)

# +
# Using the best-fit SVM model to predict on the test set
svm_clf = SVC(C=10000.0, gamma=0.001, kernel='rbf', class_weight='balanced')
svm_clf.fit(train_vectors, train_labels)
predicted_labels = svm_clf.predict(test_vectors)

# Calculating and printing the confusion matrix
conf_mat = confusion_matrix(test_labels, predicted_labels)
#print("Confusion matrix:\n", conf_mat)
ConfusionMatrixDisplay.from_estimator(svm_clf, test_vectors, test_labels)
# Calculating precision, recall, and accuracy
tp = conf_mat[1, 1]
fp = conf_mat[0, 1]
fn = conf_mat[1, 0]
tn = conf_mat[0, 0]
precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
# -

# ### Final classifier that is used for prediction

# Using the best-fit SVM model to predict on the test set
svm_clf = SVC(C=10000.0, gamma=0.001, kernel='rbf', class_weight='balanced')
svm_clf.fit(train_vectors, train_labels)

# ### Set the predictions for each user using trained ML model

for user in mod.get_users():
    
    feature_vectors = user.get_properties()
    # Predict the label of the new instance using the trained SVM model
    predicted_label = svm_clf.predict([feature_vectors])[0]

    user.set_next_state(predicted_label)
