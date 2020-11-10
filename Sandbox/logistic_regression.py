# -*- coding: utf-8 -*-
"""
Testing the Logistic Regression Model
- Comparing inputs:
    - tweet + glove embeddings vs.
    - tweet metadata
- Comparing how increasing the number of iterations affects performance
"""
import time
import sys
import numpy as np
import pickle
from datetime import datetime
from numpy import random
import csv
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten, Dense
import matplotlib as mpl
import matplotlib.pyplot as plt

path_to_m4rdata = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"


# Importing load_italian_data and running it:
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Sandbox")
from load_italian_data import *

# Only really need to run the following code once as a kind of setup:
# Or can skip this because ive just pickled the tensors!

# Have to limit the size of the t_samples
"""
a = time.time()
# Loading the training data:
t_samples, t_labels, t_metadata = load_training_data()
# Loading the validation data:
v_samples, v_labels, v_metadata = load_validation_data()
# Creating the vectorizer (this will convert the text to a list of integers)
vectorizer = make_vocab_index(t_samples, output_sequence_length = 100)
# Making class names
class_names = ["human", "bot"]
# Loading the GloVe embeddings (Twitter corpus)
embeddings_index = load_glove_embeddings()
# Creating the embedding matrix (based on the tokens deemed important by the vectorizer)
embedding_matrix = create_embedding_matrix(embeddings_index, vectorizer)
# Specifiying num_tokens:
num_tokens = len(vectorizer.get_vocabulary()) + 2
# Creating the embedding layer
embedding_layer = create_embedding_layer(embedding_matrix,num_tokens)
# Tensorfying the validation and training samples
ni = 2000000
X_t, X_v, y_t, y_v = create_tensor_slices(t_samples[:ni], v_samples, t_labels[:ni], v_labels, vectorizer)

Z_t = tf.convert_to_tensor(t_metadata[:ni])
Z_v = tf.convert_to_tensor(v_metadata)
#b = time.time()
#print("Took", b-a, "seconds to run.")
"""

# Saving the tensors:
"""
pickle.dump(X_t, open(path_to_m4rdata+"italian_x_t.p", "wb" ))
pickle.dump(X_v, open(path_to_m4rdata+"italian_x_v.p", "wb" ))
pickle.dump(y_t, open(path_to_m4rdata+"italian_y_t.p", "wb" ))
pickle.dump(y_v, open(path_to_m4rdata+"italian_y_v.p", "wb" ))
pickle.dump(Z_t, open(path_to_m4rdata+"italian_z_t.p", "wb" ))
pickle.dump(Z_v, open(path_to_m4rdata+"italian_z_v.p", "wb" ))
"""
# Loading them:
"""
X_t = pickle.load(open(path_to_m4rdata+"italian_x_t.p", "rb" ))
X_v = pickle.load(open(path_to_m4rdata+"italian_x_v.p", "rb" ))
y_t = pickle.load(open(path_to_m4rdata+"italian_y_t.p", "rb" ))
y_v = pickle.load(open(path_to_m4rdata+"italian_y_v.p", "rb" ))
Z_t = pickle.load(open(path_to_m4rdata+"italian_z_t.p", "rb" ))
Z_v = pickle.load(open(path_to_m4rdata+"italian_z_v.p", "rb" ))
"""

def run1():
    """
    Creates and trains an lbfgs logistic regressor on tweet data
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    # Times how long it took to run (roughly about half an hour)
    """
    
    solver = 'lbfgs'
    max_iter = 100000
    a = time.time()
    clf = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(X_t, y_t)
    b = time.time()
    print("Training complete after", b-a, "seconds.")
    return clf    

def run2():
    """
    Creates and trains an lbfgs logistic regressor on metadata
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    # Times how long it took to run (roughly about half an hour)
    """
    
    solver = 'lbfgs'
    max_iter = 100000
    a = time.time()
    clf = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(Z_t, y_t)
    b = time.time()
    print("Training complete after", b-a, "seconds.")
    return clf 

"""
ni  = 2000000
t_probas = clf.predict_proba(X_t)
v_probas = clf.predict_proba(X_v)
new_t_metadata = []
new_v_metadata = []
for x, y in zip(t_metadata[:ni], t_probas):
    new_t_metadata.append(np.append(x,y))
for x, y in zip(v_metadata[:ni], v_probas):
    new_v_metadata.append(np.append(x,y)) 
"""

def run3():
    """
    Tries to train a Logistic Regressor on Tweets and Metadata
    """
    solver = 'lbfgs'
    max_iter = 100000
    a = time.time()
    clf = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(new_t_metadata, y_t)
    b = time.time()
    print("Training complete after", b-a, "seconds.")
    return clf 

# To test:
# clf.score(X_t,y_t) or clf.score(X_v, y_v)
    
