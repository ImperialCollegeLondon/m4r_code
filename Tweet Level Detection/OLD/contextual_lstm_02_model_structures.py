# Modules
import sys, pickle
import tensorflow as tf
import pandas as pd
import numpy as np
# Importing functions to create Contextual LSTM layers:
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Concatenate
# Other things to load:
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import cast
# File paths
path_to_m4rdata = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
path_to_data_as_pandas = path_to_m4rdata+"Italian Data as Pandas\\"
path_to_glove_file = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\GloVe\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
path_to_m4r_code_tld = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection\\"


def contextual_lstm_model_1():
    """
    LOADS MODEL
    Trying to construct a single pipeline:
    So we "name" each layer, and using () after the function calls the layer previous:
    ISN'T WORKING: COMING UP WITH THE ERROR:
    'Dense' object has no attribute 'op'
    ***** Is working now!!
    # ADJUST HYPERPARAMETERS AND STRUCTURES!
    embed_layer with 1000 out seems to be much better than with 100 out!
    accuracy: 0.7550 (1000) vs 0.6900 (100)
    """
    
    # i.e. max_length = max_features
    # max_length might need to be 50002
    max_length = 50000 # maximum (different) tokens we can have
    tweet_input = Input(shape=(max_length,), dtype = 'int64', name = "tweet_input")
    embed_layer = Embedding(max_length, 100, trainable = False)(tweet_input)
    lstm_layer = LSTM(100)(embed_layer)
    auxiliary_output = Dense(1, activation="sigmoid", name = "auxiliary_output")(lstm_layer) # side output, won't contribute to the contextual LSTM, just so we can see what the LSTM is doing/have predicted
    metadata_input = Input(shape = (6,), name = "metadata_input") # there are 6 pieces of metadata
    new_input = Concatenate(axis=-1)([cast(lstm_layer, "float32"), cast(metadata_input, "float32") ])
    hidden_layer_1 = Dense(128, activation = "relu")(new_input)
    hidden_layer_2 = Dense(128, activation = "relu")(hidden_layer_1)
    final_output = Dense(1, activation = "sigmoid", name = "final_output")(hidden_layer_2)
    
    
    # compiling the model:
    model = Model(inputs = [tweet_input, metadata_input], outputs = [final_output, auxiliary_output])
    model.compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = ["acc"],
                 loss_weights = [0.8, 0.2])
    model.summary()
    return model













