"""
Created on Wed Feb 17 07:39:53 2021

@author: fangr
"""

# *- MODULES -*
import sys, pickle
import pandas as pd
import numpy as np
import datetime
import time
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# *- FILE PATHS -*
indiana = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Indiana Dataset\\"
italy   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Cresci Italian Dataset\\"
lstmdata= "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\LSTM Training Data\\"
dset = ["astroturf.p", "cresci-rtbust-2019.p", "gilani-2017.p", "midterm_2018.p", "political-bots-2019.p", "social_spambots_1.p", "social_spambots_2.p", "social_spambots_3.p", "traditional_spambots_1.p", "varol-2017.p", "genuine_accounts.p", "gilani-2017.p"]
dset = ["bot_sample_" + dset[x] if x < 10 else "human_sample_" + dset[x] for x in range(12) ]
glove_file = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\glove.twitter.27B\\glove.twitter.27B.100d.txt"


def load_training_data():
    # Constructing a giant dataset:
    df = pd.DataFrame()
    for i in range(12):
        d = pickle.load(open(lstmdata + dset[i], "rb"))
        df = pd.concat([df,d] , ignore_index = True)
    df = df.sample(frac = 1, random_state = 9*1349565)
    df = df.fillna(0)
    return df

def tensorfy(df, max_tokens = 50000, output_sequence_length = 40, standardize = None, batch_size = 128, save = False):
    """
    Note: (using df["tweet"].str.len().quantile(q = 0.95) )
        90% of tweets have <30 tokens
        95% of tweets have <34 tokens
        98% of tweets have <43 tokens
        99% of tweets have <51 tokens
        
        97% of tweets have <40 tokens
    Note: (using df["tweet"].str.len().describe() )
        Average number of tokens is 16
    """
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length, standardize = standardize)
    text_to_adapt_to = tf.data.Dataset.from_tensor_slices(df["tweet"].str.join(" ").tolist()).batch(batch_size)
    vectorizer.adapt(text_to_adapt_to)
    
    # Vectorizing training data in batches
    vectorized_training_data_batches = []
    r = int(np.ceil(len(df)/128))
    print(" ")
    for i in range(r):
        sys.stdout.write("\r" + str(i+1) + " out of " + str(r))
        vectorized_training_data_batches.append(vectorizer(df["tweet"].str.join(" ")[128*i:128*(i+1)].tolist()))
    trn_tweet_vector = tf.concat(vectorized_training_data_batches, axis = 0)
    print(" ")
    # Retrieving metadata
    trn_metadata = df.iloc[:,1:22].to_numpy()
    
    # Retrieving labels
    trn_labels = df["label"].to_numpy().astype(int)
    
    # Save?
    if save:
        pickle.dump(trn_tweet_vector, open(lstmdata + "trn_tweet_vector.p", "wb"))
        pickle.dump(trn_metadata, open(lstmdata + "trn_metadata.p", "wb"))
        pickle.dump(trn_labels, open(lstmdata + "trn_labels.p", "wb"))
        pickle.dump(vectorizer, open(lstmdata + "vectorizer.p", "wb"))        
    
    return vectorizer, trn_tweet_vector, trn_metadata, trn_labels

def create_embeddings_index():
    """
    Loading the twitter glove file 100d (100 dimensions)
    *DIMENSIONS = 100, IF THIS WERE TO CHANGE, CHANGE THE IF STATEMENT*
    """
    embeddings_index = {}
    with open(glove_file,'r',encoding = "utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype = "f", sep=" ")
            if len(coefs) == 100:# checking to see it matches the dimension!
                embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index

def create_embedding_matrix(vectorizer, embeddings_index):
    """
    Creates the embedding_matrix
    NOTE: num_tokens = len(voc) + 2 (the +2 is for padding of "[UNK]" (unknown) and "" (empty space))
    NOTE: If dimensions change, change embedding_dim (currently set to 100)
    """
    hits = 0; misses = 0 # variables to record number of words converteds
    # retrieving voc and word_index from vectorizer
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    # Shape of the embedding matrix:
    num_tokens = len(voc) + 2
    embedding_dim = 100
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():# iterating through the items in the vectorizer
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
    

