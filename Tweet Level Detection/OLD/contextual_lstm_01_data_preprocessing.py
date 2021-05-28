# Modules
import sys, pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# File paths
path_to_m4rdata = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
path_to_glove_file = path_to_m4rdata + "glove.twitter.27B\\glove.twitter.27B.100d.txt"
path_to_italy = path_to_m4rdata + "Cresci Italian Dataset\\"

def s00_load_data():
    """
    Loading the pickled pandas dataframe of the labelled Italian dataset
    """
    dfg = pickle.load(open(path_to_italy + "pandas_genuine_accounts_tweets.p", "rb"))
    dfg["bot"] = np.zeros((len(dfg)), dtype = int)
    dfg["dataset"] = np.zeros((len(dfg)))
    dfs1 =  pickle.load(open(path_to_italy + "pandas_social_spambots_1_tweets.p", "rb"))
    dfs1["bot"] = np.ones((len(dfs1)), dtype = int)
    dfs1["dataset"] = np.ones((len(dfs1)))
    dfs2 =  pickle.load(open(path_to_italy + "pandas_social_spambots_2_tweets.p", "rb"))
    dfs2["bot"] = np.ones((len(dfs2)), dtype = int)
    dfs2["dataset"] = 2.*np.ones((len(dfs2)))
    dfs3 =  pickle.load(open(path_to_italy + "pandas_social_spambots_3_tweets.p", "rb"))
    dfs3["bot"] = np.ones((len(dfs3)), dtype = int)
    dfs3["dataset"] = 3.*np.ones((len(dfs3)))
    dft1 =  pickle.load(open(path_to_italy + "pandas_traditional_spambots_1_tweets.p", "rb"))
    dft1["bot"] = np.ones((len(dft1)), dtype = int)
    dft1["dataset"] = 4.* np.ones((len(dft1)))
    df_entire = pd.concat([dfg, dfs1, dfs2, dfs3, dft1])
    pickle.dump(df_entire, open(path_to_italy + "pandas_italian_dataset.p", "wb"))
    return df_entire


# When loading the genuine data, because its so large, try using the following parameters with pandas.read_csv()
# skiprows
# nrows

def s001_shuffle_data_sample_spambots(n = 400000):
    """
    n = 500,000
    Sampling n number of tweets from each spambot dataset
    Sampling 4*n number of tweets from the genuine tweets dataset
    Adding the bot label and the group label
    Then producing one giant dataset (for training purposes)
    """
    dfs1 =  pickle.load(open(path_to_italy + "pandas_social_spambots_1_tweets.p", "rb"))
    dfs1 = dfs1.sample(n)
    dfs1["bot"] = np.ones((len(dfs1)), dtype = int)
    dfs1["dataset"] = np.ones((len(dfs1)))
    # =========================================================================
    dfs2 =  pickle.load(open(path_to_italy + "pandas_social_spambots_2_tweets.p", "rb"))
    dfs2 = dfs2.sample(n)#, replace =  True)
    dfs2["bot"] = np.ones((len(dfs2)), dtype = int)
    dfs2["dataset"] = 2.*np.ones((len(dfs2)))
    # =========================================================================
    dfs3 =  pickle.load(open(path_to_italy + "pandas_social_spambots_3_tweets.p", "rb"))
    dfs3 = dfs3.sample(n)
    dfs3["bot"] = np.ones((len(dfs3)), dtype = int)
    dfs3["dataset"] = 3.*np.ones((len(dfs3)))
    # =========================================================================
    dft1 =  pickle.load(open(path_to_italy + "pandas_traditional_spambots_1_tweets.p", "rb"))
    dft1 = dft1.sample(n, replace = True)
    dft1["bot"] = np.ones((len(dft1)), dtype = int)
    dft1["dataset"] = 4.* np.ones((len(dft1)))
    # =========================================================================
    df_spambots = pd.concat([dfs1, dfs2, dfs3, dft1]).sample(frac = 1.0).reset_index(drop = True)
    return df_spambots
    

def s002_shuffle_data_sample_genuine(n = 4*400000):
    """
    Sampling the 4*n number of tweets 

    """
    dfg = pickle.load(open(path_to_italy + "pandas_genuine_accounts_tweets.p", "rb"))
    dfg = dfg.sample(n)
    dfg["bot"] = np.zeros((len(dfg)), dtype = int)
    dfg["dataset"] = np.zeros((len(dfg)))
    return dfg
    
    
def s003_shuffled_data_tensorfying(max_tokens = 50000, output_sequence_length = 50, standardize = None, batch_size = 128):
    df = pickle.load(open(path_to_italy + "pandas_3200000_stratified_shuffled.p", "rb"))
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length, standardize = standardize)
    text_to_adapt_to = tf.data.Dataset.from_tensor_slices(df["text"].tolist()).batch(batch_size)
    vectorizer.adapt(text_to_adapt_to)
    
    # Vectorizing training data in batches
    vectorized_training_data_batches = []
    for i in range(int(np.ceil(len(df)/128))):
        vectorized_training_data_batches.append(vectorizer(df["text"][128*i:128*(i+1)].tolist()))
    trn_tweet_vector = tf.concat(vectorized_training_data_batches, axis = 0)
    
    training_metadata = df[["retweet_count", "reply_count", "favorite_count", "num_hashtags", "num_urls", "num_mentions"]].to_numpy()
    
    training_labels = df["bot"].to_numpy()
    
    pickle.dump(trn_tweet_vector, open(path_to_italy + "training_tweets_tensor.p", "wb"))
    pickle.dump(training_metadata, open(path_to_italy + "training_metadata.p", "wb"))
    pickle.dump(training_labels,  open(path_to_italy + "training_labels.p", "wb"))
    
    # df = df[["tweet_id"]]
    # pickle.dump(df, open(path_to_italy + "pandas_LSTM_training_data.p","wb"))
    
def s004_():
    
    pass











def s01_split_data(df_entire, n = 2000000, split = 0.8):
    """
    Splitting the labelled dataset into training and validation dataframes
    """
    a = df_entire[df_entire["bot"]==0].sample(n)
    b = df_entire[df_entire["bot"]==1].sample(n, replace = True)
    
    df_trn = pd.concat([a.iloc[:int(n*split),:], b.iloc[:int(n*split),:]]).sample(frac=1.0)
    df_val = pd.concat([a.iloc[int(n*split):,:], b.iloc[int(n*split):,:]]).sample(frac=1.0)
    return df_trn, df_val



def s02_make_vectorizer(df_trn, max_tokens = 50000, output_sequence_length = 50, standardize = None, batch_size = 128):
    """
    Making the vectorizer (text => vector of integers)
    * I've had to change the following file: 'C:/Users/fangr/anaconda3/lib/site-packages/tensorflow/python/keras/layers/preprocessing/string_lookup.py'
    Inputs:
        max_tokens = maximum number of different tokens that we want * might be too many at the moment? may result in too many neurons or smthn?
        output_sequence_length = number of tokens to pad to - enforces a strict shape on the output tensor (e.g. if the tweet only has 10 characters, the remaining padding will be all zeros)
        output_sequence_length = Number of tokens to pad to - enforces a strict shape on the output tensor (e.g. if the tweet only has 10 characters, the remaining padding will be all zeros)
        standardize = don't need to standardise because we've already tokenised the tweets, and tokenising removes all of the capital letters
        batch_size = what batch size for training for each step in the LSTM
    """
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length, standardize = standardize)
    text_to_adapt_to = tf.data.Dataset.from_tensor_slices(df_trn["text"].tolist()).batch(batch_size)
    vectorizer.adapt(text_to_adapt_to)
    return vectorizer

def s03_embeddings_index():
    """
    Loading the twitter glove file 100d (100 dimensions)
    *DIMENSIONS = 100, IF THIS WERE TO CHANGE, CHANGE THE IF STATEMENT*
    """
    embeddings_index = {}
    with open(path_to_glove_file,'r',encoding = "utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype = "f", sep=" ")
            if len(coefs) == 100:# checking to see it matches the dimension!
                embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index

def s04_embedding_matrix(vectorizer, embeddings_index):
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


















