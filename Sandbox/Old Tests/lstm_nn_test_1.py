"""
Implementing LSTM on the Italian Dataset in smaller data batch sizes
"""

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


# Loading the Pickled Italian dataset (this has been shuffled and split already)

def load_training_data(n = None):
    train_samples  = pickle.load(open(path_to_m4rdata+"italian_train_samples.p", "rb" ))
    train_labels   = pickle.load(open(path_to_m4rdata+"italian_train_labels.p", "rb" ))
    train_metadata = pickle.load(open(path_to_m4rdata+"italian_train_metadata.p", "rb" )) 
    if n == None:
        return train_samples, train_labels, train_metadata
    else:
        return train_samples[:n], train_labels[:n], train_metadata[:n]

def load_validation_data(n = None):
    val_samples    = pickle.load(open(path_to_m4rdata+"italian_val_samples.p", "rb" ))
    val_labels     = pickle.load(open(path_to_m4rdata+"italian_val_labels.p", "rb" ))
    val_metadata   = pickle.load(open(path_to_m4rdata+"italian_val_metadata.p", "rb" ))
    if n == None:
        return val_samples, val_labels, val_metadata
    else:
        return val_samples[:n], val_labels[:n], val_metadata[:n]








# Preparing the samples, and the GloVe embeddings

def make_vocab_index(train_samples, max_tokens = 10000, batch_size = 128, output_sequence_length = None):
    """
    Creates tensor slices
    Creates vectorizer and adapts it to the training samples
    """
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length, standardize = None)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(batch_size)
    vectorizer.adapt(text_ds)

    return vectorizer

def load_glove_embeddings():
    """
    *DIMENSIONS = 100, IF THIS WERE TO CHANGE, CHANGE THE IF STATEMENT*
    Loads the twitter glove file 100d
    """
    path_to_glove_file = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\GloVe\\glove.twitter.27B\\glove.twitter.27B.100d.txt"

    embeddings_index = {}
    with open(path_to_glove_file,'r',encoding = "utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype = "f", sep=" ")
            if len(coefs) == 100:
                embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    return embeddings_index


def create_embedding_matrix(embeddings_index, vectorizer):
    """
    Creates embedding_matrix
    NOTE:
    num_tokens = len(voc) + 2
    (the +2 is for padding of [UNK] and "")
    NOTE:
    If dimensions change, change embedding_dim (currently set to 100)
    """
    # int variables to record number of words converted
    hits = 0
    misses = 0

    # retrieving voc and word_index from vectorizer
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    # Shape of the embedding matrix:
    num_tokens = len(voc) + 2
    embedding_dim = 100

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix

def create_embedding_layer(embedding_matrix, num_tokens, embedding_dim = 100):
    """
    trainable = False
    ! to ensure that the model doesn't try and change the PRETRAINED GloVe embeddings!
    num_tokens = maximum number of unique tokens to look for?
    """
    embedding_layer = Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    )

    return embedding_layer

def create_tensor_slices(train_samples, val_samples, train_labels, val_labels, vectorizer):
    x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
    x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    """
    *Pretty sure this does the same thing as sequence.pad_sequences*
    define x_train by changing words/tokens to integers
    input_train = sequence.pad_sequences(x_train, maxlen = 500, padding = "post")
    """
    return x_train, x_val, y_train, y_val

def prepare_lstm_100():
    """
    Creates and trains a logistic regressor on 100 random samples
    Validates the logistic regressor on 100 random validation samples
    ----------------------------------------------------
    Training accuracy: 0.81
    Validation accuracy: 0.55
    """
    # Loading the first 100 items of the training data:
    t_samples, t_labels, t_metadata = load_training_data(100)
    # Loading the *entire* validation data:
    v_samples, v_labels, v_metadata = load_validation_data(1000000)
    # Creating the vectorizer (this will convert the text to a list of integers)
    # Standardizing the output sequence length to be 50 to ensure that each
    # element of t_samples and v_samples has the same shape
    # e.g. if output_sequence_length = None, then the vectorizer will vectorize
    # the separate corpuses and the length of each will be the maximum number
    # of tokens in a tweet in each corpus
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
    X_t, X_v, y_t, y_v = create_tensor_slices(t_samples, v_samples, t_labels, v_labels, vectorizer)
    
    return X_t, X_v, y_t, y_v, num_tokens

def lstm_100():
    
    """
    max_features = len(vectorizer.get_vocabulary()) + 2
    print("max_features =", max_features)
    model_1 = Sequential()
    model_1.add(Embedding(max_features, 32))
    model_1.add(LSTM(units = 32))
    model_1.add(Dense(1, activation = "sigmoid"))
    model_1.compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = ["acc"])
    history_1 = model_1.fit(X_t, y_t, epochs=10, batch_size=128, validation_split=0.2)
    """
    
    """
    # Increased LSTM Units
    max_features = len(vectorizer.get_vocabulary()) + 2
    print("max_features =", max_features)
    model_2 = Sequential()
    model_2.add(Embedding(max_features, 100))
    model_2.add(LSTM(units = 100))
    model_2.add(Dense(1, activation = "sigmoid"))
    model_2.compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = ["acc"])
    history_2 = model_2.fit(X_t, y_t, epochs=10, batch_size=128, validation_split=0.2)
    """
    
    # Increased LSTM units, changed activation function, increased number of epochs
    max_features = num_tokens #500 #len(vectorizer.get_vocabulary()) + 2
    print("max_features =", max_features)
    model_3 = Sequential()
    model_3.add(Embedding(max_features, 100))
    model_3.add(LSTM(units = 100))
    model_3.add(Dense(1, activation = "tanh"))
    model_3.compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = ["acc"])
    history_3 = model_3.fit(X_t, y_t, epochs=30, batch_size=128, validation_split=0.2)
    
    return history_3, model_3