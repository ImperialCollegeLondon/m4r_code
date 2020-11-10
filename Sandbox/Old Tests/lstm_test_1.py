"""
LSTM Bot detector for caverlee_2011
* Notes:
With successive epochs, the Accuracy score remained about 0.5808 with no change
10 epochs would have taken around 4 hours
There was also an issue with the size of the tensors,
especially the training sample tensors, so the training set was reduced
"""
# Importing modules
import sys
import numpy as np
import pickle
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

# File Paths
path_to_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"

# STRUCTURE OF FILES HAS CHANGED!

def load_samples():
    """
    Loads the pickled caverlee_2011 samples and labels
    """
    samples = pickle.load(open(path_to_data + "caverlee_2011_samples.p", "rb"))
    labels = pickle.load(open(path_to_data + "caverlee_2011_labels.p", "rb"))
    return samples, labels

def split_samples(samples, labels, seed = 1337, validation_split = 0.2):
    """
    Shuffles and splits into training and validation sets
    Could alternatively use:  train_test_split from sklearn.model_selection
    """
    # *shuffling*
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)

    # *splitting*
    num_validation_samples = int(validation_split * len(samples))
    train_samples = samples[:-num_validation_samples]
    val_samples = samples[-num_validation_samples:]
    train_labels = labels[:-num_validation_samples]
    val_labels = labels[-num_validation_samples:]

    return train_samples, val_samples, train_labels, val_labels

def make_vocab_index(train_samples, max_tokens = 10000, batch_size = 128, output_sequence_length = None):
    """
    Creates tensor slices
    Creates vectorizer and adapts it to the training samples
    """
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=None, standardize = None)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(batch_size)
    vectorizer.adapt(text_ds)

    return vectorizer

class_names = ["human", "bot"]

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

def build_lstm_model(max_features):
    """
    max_features = max_tokens
    max_tokens = len(vectorizer.get_vocabulary())
    Attempt at building an lstm model based on imdb model
    https://github.com/LJANGN/Analysing-IMDB-reviews-using-GloVe-and-LSTM/blob/master/WordEmbeddingsWithKeras.ipynb
    """
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation = "sigmoid"))
    
    model.compile(optimizer = "rmsprop",
                 loss = "binary_crossentropy",
                 metrics = ["acc"])
    
    return model

def train_lstm():
    """
    Order of things to run in:
    """
    #1.
    samples, labels = load_samples()
    train_samples, val_samples, train_labels, val_labels = split_samples(samples, labels, validation_split = 0.25)
    vectorizer = make_vocab_index(train_samples, max_tokens = 20000, output_sequence_length = 200)
    x_train, x_val, y_train, y_val = create_tensor_slices(train_samples, val_samples, train_labels, val_labels, vectorizer)
    # clears variables to make room for memory?
    train_samples = []; val_samples = []; train_labels = []; val_labels = []
    samples = []; labels = []
    embeddings_index = load_glove_embeddings()
    embedding_matrix = create_embedding_matrix(embeddings_index, vectorizer)
    embedding_layer = create_embedding_layer(embedding_matrix, 20000)
    model = build_lstm_model(20000)
    history = model.fit(X_t, y_t, epochs=30, batch_size=128, validation_split=0.2)
    return history, model