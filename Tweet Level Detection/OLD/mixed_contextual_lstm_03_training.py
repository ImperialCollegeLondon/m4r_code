"""
Created on Wed Feb 17 07:39:53 2021

@author: fangr
"""

# *- MODULES -*
import sys, pickle
import pandas as pd
import numpy as np
import time

# Importing functions to create Contextual LSTM layers:
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.initializers import Constant
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import cast


# Local modules
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection")
from mixed_contextual_lstm_02_preprocessing import *


# *- FILE PATHS -*
indiana = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Indiana Dataset\\"
italy   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Cresci Italian Dataset\\"
lstmdata= "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\LSTM Training Data\\"
dset = ["astroturf.p", "cresci-rtbust-2019.p", "gilani-2017.p", "midterm_2018.p", "political-bots-2019.p", "social_spambots_1.p", "social_spambots_2.p", "social_spambots_3.p", "traditional_spambots_1.p", "varol-2017.p", "genuine_accounts.p", "gilani-2017.p"]
dset = ["bot_sample_" + dset[x] if x < 10 else "human_sample_" + dset[x] for x in range(12) ]
glove_file = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
weightsavepath = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection\\mixed_contextual_LSTM_weights_1\\"


def mixed_contextual_lstm_model(embed_mat):
    """
    Defines LSTM model structure
    """
    
    # i.e. max_length = max_features
    # max_length might need to be 50002
    # max_length = 50000 # maximum (different) tokens we can have
    max_length = 40 # = output_sequence_length
    tweet_input = Input(shape=(max_length,), dtype = 'int64', name = "tweet_input")
    embed_layer = Embedding(50002, 100, embeddings_initializer=Constant(embed_mat), input_length = 40, trainable = False)(tweet_input)
    lstm_layer = LSTM(100)(embed_layer)
    auxiliary_output = Dense(1, activation="sigmoid", name = "auxiliary_output")(lstm_layer) # side output, won't contribute to the contextual LSTM, just so we can see what the LSTM is doing/have predicted
    metadata_input = Input(shape = (21,), name = "metadata_input") # there are 21 pieces of metadata
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

def mixed_contextual_lstm_model_old():
    """
    Defines LSTM model structure
    possibly incorrect because the embedding layer is incorrect?
    """
    
    # i.e. max_length = max_features
    # max_length might need to be 50002
    max_length = 50000 # maximum (different) tokens we can have
    tweet_input = Input(shape=(max_length,), dtype = 'int64', name = "tweet_input")
    embed_layer = Embedding(max_length, 100, trainable = False)(tweet_input)
    lstm_layer = LSTM(100)(embed_layer)
    auxiliary_output = Dense(1, activation="sigmoid", name = "auxiliary_output")(lstm_layer) # side output, won't contribute to the contextual LSTM, just so we can see what the LSTM is doing/have predicted
    metadata_input = Input(shape = (21,), name = "metadata_input") # there are 21 pieces of metadata
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

def loading_data():
    print("Loading Data")
    a = time.time()
    df = load_training_data()
    b = time.time()
    sys.stdout.write(" (" + str(b-a) +" seconds)")
    print("\nCreating vectorizer")
    a = time.time()
    vectorizer, trn_tweet_vector, trn_metadata, trn_labels = tensorfy(df)
    b = time.time()
    sys.stdout.write(" (" + str(b-a) +" seconds)")
    print("\nCreating embeddings index")
    a = time.time()
    embeddings_index = create_embeddings_index()
    b = time.time()
    sys.stdout.write(" (" + str(b-a) +" seconds)")
    print("\nCreating embedding matrix")
    a = time.time()
    embedding_matrix = create_embedding_matrix(vectorizer, embeddings_index)
    b = time.time()
    sys.stdout.write("\n(" + str(b-a) +" seconds)")
    return vectorizer, trn_tweet_vector, trn_metadata, trn_labels, embeddings_index, embedding_matrix


def train_mixed_model(num_epochs = 10, batch_size = 64, load_weights = None, vectorizer = None, trn_tweet_vector = None, trn_metadata = None, trn_labels = None, embeddings_index = None, embedding_matrix = None):
    # ====================
    # Check point:
    # ====================
    # Model weights will be saved if they are the best seen so far (in terms of validation accuracy)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weightsavepath, save_weights_only=True, # can set this to false but would then need to use load_model instead of load_weights
        save_freq = "epoch",
        monitor='final_output_acc',
        mode='max',
        save_best_only=True)
    
    # ====================
    # Load Data etc
    # Takes ages to run the vectorizer so maybe just load it once and save and store
    # ====================
    if vectorizer == None:
        vectorizer, trn_tweet_vector, trn_metadata, trn_labels, embeddings_index, embedding_matrix = loading_data()
    
    # ====================
    # Loading the model structure:
    # ====================
    print("Loading model structure / weights")
    # model = mixed_contextual_lstm_model()
    model = mixed_contextual_lstm_model(embedding_matrix)
    
    if load_weights != None:
        # ====================
        # loading previous model weights:
        # ====================
        model.load_weights(weightsavepath)
    
    
    # ====================
    # TRAINING
    # ====================
    print("Begin training...")
    # Training the model on very few epochs
    split = int(0.8*len(trn_tweet_vector))
    history = model.fit(
        {'tweet_input': trn_tweet_vector[:split,:], 'metadata_input': trn_metadata[:split,:]},
        {'final_output': trn_labels[:split], 'auxiliary_output': trn_labels[:split]},
        validation_data = (
            {'tweet_input': trn_tweet_vector[split:, :], 'metadata_input': trn_metadata[split:,:]},
            {'final_output': trn_labels[split:], 'auxiliary_output': trn_labels[split:]}
            ),
        epochs = num_epochs,
        batch_size = batch_size,
        callbacks=[model_checkpoint_callback]
        )
    return model, history


