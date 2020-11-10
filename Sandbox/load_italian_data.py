# -*- coding: utf-8 -*-
"""
Testing the Logistic Regression Model
- Comparing inputs:
    - tweet + glove embeddings vs.
    - tweet metadata
- Comparing how increasing the number of iterations affects performance
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


