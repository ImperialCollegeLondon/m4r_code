"""
Extrapolating from the keras tutorial
"""
import sys
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras import layers
import numpy as np
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten, Dense

# Deleting the first, second, and last column of the truncated caverlee_2011 datasets
# Tokenising the tweets
# Adding to 'samples'
path_to_sandbox = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Sandbox\\"

def creating_samples():
    """
    Loads the truncated caverlee datasets, tokenises them,
    and pickles the samples and labels retrieved
    """
    samples_humans = []
    samples_bots = []
    labels = []
    with open(path_to_sandbox+"1500_samples_humans_caverlee_2011_1.txt", 'r', encoding = "utf-8") as r:
        for i, line in enumerate(r):
            tweet = line.split("\t")[2:-1]
            tweet = " ".join(tweet)
            tokens = text_tokeniser(tweet)
            s = " ".join(tokens)
            samples_humans.append(s)

    with open(path_to_sandbox+"1500_samples_bots_caverlee_2011_1.txt", 'r', encoding = "utf-8") as r:
        for i, line in enumerate(r):
            tweet = line.split("\t")[2:-1]
            tweet = " ".join(tweet)
            tokens = text_tokeniser(tweet)
            s = " ".join(tokens)
            samples_bots.append(s)

    samples = samples_humans + samples_bots
    labels = [0]*len(samples_humans) + [1] * len(samples_bots)
    pickle.dump(samples, open(path_to_sandbox+"1500_samples_caverlee_2011_1.p", "wb"))
    pickle.dump(labels, open(path_to_sandbox+"1500_labels_caverlee_2011_1.p", "wb"))
    # wb = write binary


def load_samples():
    """
    Loads the pickled samples and labels
    """
    samples = pickle.load(open(path_to_sandbox+"1500_samples_caverlee_2011_1.p", "rb"))
    labels = pickle.load(open(path_to_sandbox+"1500_labels_caverlee_2011_1.p", "rb"))
    return samples, labels

# samples, labels = load_samples()


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





def make_vocab_index(train_samples, max_tokens = 10000, batch_size = 128):
    """
    Creates tensor slices
    Creates vectorizer and adapts it to the training samples
    """
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=None, standardize = None)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(batch_size)
    vectorizer.adapt(text_ds)

    return vectorizer


# voc = vectorizer.get_vocabulary()
# word_index = dict(zip(voc, range(len(voc))))
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



def build_nn_model():
    """
    """
    int_sequences_input = keras.Input(shape=(None,), dtype="int64")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(int_sequences_input, preds)
    model.summary()
    model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
    )
    return model


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
    
    
    
    
    











# Some tests to do:
"""
To find out the most common appearing tokens:
    f_vectorizer.get_vocabulary()[:20]
Possibly use standardize = None to stop the textvectorization function from stripping punctuation etc, i.e. <user> -> user
the data should already be tokenised from test_tokeniser
    f_vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=200, standardize = None)
Other tests:


    output = f_vectorizer([["the cat sat on the mat"]])
    output.numpy()[0, :6]

    test = ["the", "cat", "sat", "on", "the", "mat"]
    [f_word_index[w] for w in test]


    Note: [UNK] signifies unknown?
"""
