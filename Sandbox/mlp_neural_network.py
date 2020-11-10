"""
Creates and trains a multi-layer perceptron neural network
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
from sklearn import metrics

path_to_m4rdata = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"


# Importing load_italian_data and running it:
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Sandbox")
from load_italian_data import *


# Don't need to run all of these unless console has been wiped
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
# Loading them:
"""
"""
X_t = pickle.load(open(path_to_m4rdata+"italian_x_t.p", "rb" ))
X_v = pickle.load(open(path_to_m4rdata+"italian_x_v.p", "rb" ))
y_t = pickle.load(open(path_to_m4rdata+"italian_y_t.p", "rb" ))
y_v = pickle.load(open(path_to_m4rdata+"italian_y_v.p", "rb" ))
Z_t = pickle.load(open(path_to_m4rdata+"italian_z_t.p", "rb" ))
Z_v = pickle.load(open(path_to_m4rdata+"italian_z_v.p", "rb" ))
"""

def mlp1():
    max_features = num_tokens
    model = Sequential()
    model.add(Embedding(max_features, 100))
    model.add(LSTM(100))
    model.add(Dense(1, activation = "sigmoid"))
    
    model.compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = ["acc"])

    history = model.fit(X_t, y_t, epochs=10, batch_size=128, validation_split=0.2)
    return history, model




def load_mlp1():
    """
    This loads the trained model from mlp1:
    This also loads the history of the training:
    """
    reconstructed_model = keras.models.load_model(path_to_m4rdata+"trained_lstm_nn_1")
    # Loading the training history:
    history = pickle.load(open(path_to_m4rdata+"trained_lstm_nn_1_history.p", "rb"))
    """
    print("Training Scores\n==================================")
    print("Train acc =", history["acc"])
    print("")
    # Producing predictions:
    """
    return reconstructed_model, history

def test_mlp1():
    # For Validation Scores
    y_pred = m.predict_classes(X_v)
    metrics.accuracy_score(y_v, y_pred)
    # Maybe try different metrics offered from sklearn? F1? 
    # Maybe make a heatmap?
    # For training and train validation scores:
    h['acc'][-1:]
    h['val_acc'][-1:]
    h['loss'][-1:]
    h['val_loss'][-1:]
    
    # Manually checking: use botornot/botometer and find a bot, get their tweets,
    # preprocess them, and put them through the LSTM, maybe find human tweets as well.
