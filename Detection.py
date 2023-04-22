import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import statsmodels.api as sm

df = pd.read_csv("cmse202_dataset.csv")
df

label = df["next_shared_info_prediction"]
Features = df.drop(["next_shared_info_prediction", "gender"], axis=1)
train_labels, test_labels, train_vectors, test_vectors = train_test_split(label, Features, test_size=0.25, train_size=0.75)

# +
logit_model = sm.Logit(train_labels, sm.add_constant(train_vectors))
result = logit_model.fit()

# Summarize the model
print(result.summary())
# -


