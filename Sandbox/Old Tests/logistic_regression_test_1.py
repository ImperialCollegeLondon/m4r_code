"""
Implementing Logistic Regression on the Italian Dataset
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

def logistic_regressor_100():
    """
    Creates and trains a logistic regressor on 100 random samples
    Validates the logistic regressor on 100 random validation samples
    ----------------------------------------------------
    Training accuracy: 0.81
    Validation accuracy: 0.55
    """
    # Loading the training data:
    t_samples, t_labels, t_metadata = load_training_data(100)
    # Loading the validation data:
    v_samples, v_labels, v_metadata = load_validation_data(100)
    # Creating the vectorizer (this will convert the text to a list of integers)
    # Standardizing the output sequence length to be 50 to ensure that each
    # element of t_samples and v_samples has the same shape
    # e.g. if output_sequence_length = None, then the vectorizer will vectorize
    # the separate corpuses and the length of each will be the maximum number
    # of tokens in a tweet in each corpus
    vectorizer = make_vocab_index(t_samples, output_sequence_length = 50)
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
    # Creating the logistic regressor and training it:
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    solver = 'lbfgs'
    clf = LogisticRegression(solver = solver, random_state=0).fit(X_t, y_t)
    print("Training accuracy:", clf.score(X_t, y_t))
    print("Validation accuracy:", clf.score(X_v, y_v))
    # Make heatmaps?
    return clf


def logistic_regressor_1000():
    """
    Creates and trains a logistic regressor on 1000 random samples
    Validates the logistic regressor on 1000 random validation samples
    -----------------------------------------------------------------
    Training accuracy: 0.601
    Validation accuracy: 0.533
    """
    # Loading the training data:
    t_samples, t_labels, t_metadata = load_training_data(1000)
    # Loading the validation data:
    v_samples, v_labels, v_metadata = load_validation_data(1000)
    # Creating the vectorizer (this will convert the text to a list of integers)
    # Standardizing the output sequence length to be 50 to ensure that each
    # element of t_samples and v_samples has the same shape
    # e.g. if output_sequence_length = None, then the vectorizer will vectorize
    # the separate corpuses and the length of each will be the maximum number
    # of tokens in a tweet in each corpus
    vectorizer = make_vocab_index(t_samples, output_sequence_length = 70)
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
    # Creating the logistic regressor and training it:
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    solver = 'lbfgs'
    clf = LogisticRegression(solver = solver, random_state=0).fit(X_t, y_t)
    print("Training accuracy:", clf.score(X_t, y_t))
    print("Validation accuracy:", clf.score(X_v, y_v))
    # Make heatmaps?
    return clf    

def logistic_regressor_10000():
    """
    Creates and trains a logistic regressor on 1000 random samples
    Validates the logistic regressor on 1000 random validation samples
    -----------------------------------------------------------------
    Training accuracy: 
    Validation accuracy: 
    """
    # Loading the training data:
    t_samples, t_labels, t_metadata = load_training_data(10000)
    # Loading the validation data:
    v_samples, v_labels, v_metadata = load_validation_data(10000)
    # Creating the vectorizer (this will convert the text to a list of integers)
    # Standardizing the output sequence length to be 50 to ensure that each
    # element of t_samples and v_samples has the same shape
    # e.g. if output_sequence_length = None, then the vectorizer will vectorize
    # the separate corpuses and the length of each will be the maximum number
    # of tokens in a tweet in each corpus
    vectorizer = make_vocab_index(t_samples, output_sequence_length = 70)
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
    # Creating the logistic regressor and training it:
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    solver = 'lbfgs'
    clf = LogisticRegression(solver = solver, random_state=0).fit(X_t, y_t)
    print("Training accuracy:", clf.score(X_t, y_t))
    print("Validation accuracy:", clf.score(X_v, y_v))
    # Make heatmaps?
    return clf    
    
def logistic_regressor_100000():
    """
    Creates and trains a logistic regressor on 1000 random samples
    Validates the logistic regressor on 1000 random validation samples
    -----------------------------------------------------------------
    Training accuracy: 
    Validation accuracy: 
    """
    # Loading the training data:
    t_samples, t_labels, t_metadata = load_training_data(100000)
    # Loading the validation data:
    v_samples, v_labels, v_metadata = load_validation_data(100000)
    # Creating the vectorizer (this will convert the text to a list of integers)
    # Standardizing the output sequence length to be 50 to ensure that each
    # element of t_samples and v_samples has the same shape
    # e.g. if output_sequence_length = None, then the vectorizer will vectorize
    # the separate corpuses and the length of each will be the maximum number
    # of tokens in a tweet in each corpus
    vectorizer = make_vocab_index(t_samples, output_sequence_length = 70)
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
    # Creating the logistic regressor and training it:
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    solver = 'lbfgs'
    clf = LogisticRegression(solver = solver, random_state=0).fit(X_t, y_t)
    print("Training accuracy:", clf.score(X_t, y_t))
    print("Validation accuracy:", clf.score(X_v, y_v))
    # Make heatmaps?
    return clf 
    

def logistic_regressor_1000000():
    """
    Creates and trains a logistic regressor on 1000 random samples
    Validates the logistic regressor on 1000 random validation samples
    -----------------------------------------------------------------
    Training accuracy: 
    Validation accuracy: 
    """
    # Loading the training data:
    t_samples, t_labels, t_metadata = load_training_data(1000000)
    # Loading the validation data:
    v_samples, v_labels, v_metadata = load_validation_data(1000000)
    # Creating the vectorizer (this will convert the text to a list of integers)
    # Standardizing the output sequence length to be 50 to ensure that each
    # element of t_samples and v_samples has the same shape
    # e.g. if output_sequence_length = None, then the vectorizer will vectorize
    # the separate corpuses and the length of each will be the maximum number
    # of tokens in a tweet in each corpus
    vectorizer = make_vocab_index(t_samples, output_sequence_length = 70)
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
    # Creating the logistic regressor and training it:
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    solver = 'lbfgs'
    clf = LogisticRegression(solver = solver, random_state=0).fit(X_t, y_t)
    print("Training accuracy:", clf.score(X_t, y_t))
    print("Validation accuracy:", clf.score(X_v, y_v))
    # Make heatmaps?
    return clf 

def logistic_regressor_mult_1():
    """
    Creates and trains a logistic regressor on 100, 1000, 10000, 100000,
    and 1000000 random samples
    
    Validates the logistic regressor on 1000000 random validation samples
    -----------------------------------------------------------------
    Training accuracy: 
    Validation accuracy: 
    """
    
    # Loading the training data:
    t_samples, t_labels, t_metadata = load_training_data(1000000)
    # Loading the validation data:
    v_samples, v_labels, v_metadata = load_validation_data(1000000)
    
    # Creating the vectorizer (this will convert the text to a list of integers)
    vectorizer = make_vocab_index(t_samples, output_sequence_length = 70)
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
    
    
    X_t_100, _, y_t_100 , _ = create_tensor_slices(t_samples[:100], v_samples[:1], t_labels[:100], v_labels[:1], vectorizer)
    X_t_1000, _, y_t_1000 , _ = create_tensor_slices(t_samples[:1000], v_samples[:1], t_labels[:1000], v_labels[:1], vectorizer)
    X_t_10000, _, y_t_10000 , _ = create_tensor_slices(t_samples[:10000], v_samples[:1], t_labels[:10000], v_labels[:1], vectorizer)
    X_t_100000, _, y_t_100000 , _ = create_tensor_slices(t_samples[:100000], v_samples[:1], t_labels[:100000], v_labels[:1], vectorizer)
    X_t_1000000, _, y_t_1000000 , _ = create_tensor_slices(t_samples[:1000000], v_samples[:1], t_labels[:1000000], v_labels[:1], vectorizer)
    _, X_v, _, y_v = create_tensor_slices(t_samples[:1], v_samples[:1000000], t_labels[:1], v_labels[:1000000], vectorizer)
    
    # Creating the logistic regressor and training it:
    # solver_set = {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
    solver = 'lbfgs'
    max_iter = 1000
    clf_100 = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(X_t_100, y_t_100)
    clf_1000 = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(X_t_1000, y_t_1000)
    clf_10000 = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(X_t_10000, y_t_10000)
    clf_100000 = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(X_t_100000, y_t_100000)
    clf_1000000 = LogisticRegression(solver = solver, max_iter = max_iter, random_state=0).fit(X_t_1000000, y_t_1000000)
    
    
    # Printing the training accuracies:
    print("=================================")
    print("Training accuracies")
    print("100:    ", clf_100.score(X_t_100, y_t_100))
    print("1000:   ", clf_1000.score(X_t_1000, y_t_1000))
    print("10000:  ", clf_10000.score(X_t_10000, y_t_10000))
    print("100000: ", clf_100000.score(X_t_100000, y_t_100000))
    print("1000000:", clf_1000000.score(X_t_1000000, y_t_1000000))
    
    # Printing the validation accuracies:
    print("=================================")
    print("Validation accuracies")
    print("100:    ", clf_100.score(X_v, y_v))
    print("1000:   ", clf_1000.score(X_v, y_v))
    print("10000:  ", clf_10000.score(X_v, y_v))
    print("100000: ", clf_100000.score(X_v, y_v))
    print("1000000:", clf_1000000.score(X_v, y_v))
    
    # Make heatmaps?
    return clf_100, clf_1000, clf_10000, clf_100000, clf_1000000


def logistic_regressor_mult_2():
    """
    Creates and trains a logistic regressor on 100, 1000, 10000, 100000,
    and 1000000 random samples
    
    Validates the logistic regressor on 1000000 random validation samples
    -----------------------------------------------------------------
    Training accuracy: 
    Validation accuracy: 
    """
    