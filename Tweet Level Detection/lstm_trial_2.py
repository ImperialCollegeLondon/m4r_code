# -*- coding: utf-8 -*-
"""

"""

# Modules to import
import json
import pandas as pd
import sys
import numpy as np
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf
from tensorflow import keras



# Desired GloVe dimension:
m = 25


# File paths
path_to_m4r = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\"
path_to_caverlee = path_to_m4r + "Indiana Dataset\\caverlee-2011\\social_honeypot_icwsm_2011\\"
path_to_embeddings = path_to_m4r + "GloVe\\glove.twitter.27B\\glove.twitter.27B.25d.txt"





# Loading data and splitting:
def load_data(n=None):
    """
    n = number of tokenised tweets to load from each dataset 

    """
    samples_humans = []
    samples_bots = []
    
    # GROSSLY INEFFICIENT
    # Data is NO LONGER STORED HERE
    df_humans = pd.read_csv(path_to_m4r+"m4r_code\\Data\\labelled_tokenised_caverlee_humans.csv", header=None).astype('O')
    df_bots = pd.read_csv(path_to_m4r+"m4r_code\\Data\\labelled_tokenised_caverlee_bots.csv", header=None).astype('O')
    
    df_humans.columns = ["tweets"]
    df_bots.columns = ["tweets"]
    
    # df_humans.apply(str)
    # df_bots.apply(str)
    
    df_humans.iloc[:, 0] = df_humans.iloc[:,0].apply(lambda x: eval(x))
    df_bots.iloc[:, 0] = df_bots.iloc[:,0].apply(lambda x: eval(x))    
    # df_humans.iloc[:,0]= pd.eval(df_humans.iloc[:,0], parser = "python")
    # df_bots.iloc[:,0]= pd.eval(df_bots.iloc[:,0], parser = "python")
    
    # df_humans= pd.eval(df_humans, parser = "python")
    # df_bots= pd.eval(df_bots, parser = "python")
    
    # df_humans["tweets"] = df_humans["tweets"].astype('O')
    
    # for i in range(len(df_humans)):
    #     samples_humans.append(eval(df_humans.iloc[i,0]))
        
    # for i in range(len(df_bots)):
    #     samples_bots.append(eval(df_bots.iloc[i,0]))
    
    
    
    #samples_humans = df_humans.values.tolist()
    #samples_bots = df_bots.values.tolist()
    #samples_humans = [eval(a) for a in samples_humans]
    
    # JUST NEEDED eval FUNCTION
    
    
    """
    # currently not working!
    samples_humans = []
    with open(path_to_m4r+"m4r_code\\Data\\labelled_tokenised_caverlee_humans.csv", 'r', encoding = "utf-8") as t:
        for i, line in enumerate(t):
            tweet = line
            print(tweet)
            print(type(tweet))
            tweet = eval(line)
            print(tweet)
            print(type(tweet))
            tweet = eval('"'+tweet+'"')
            print(tweet)
            print(type(tweet))
            samples_humans.append(tweet)
            if i == n:
                break
        
    
    samples_bots = []
    with open(path_to_m4r+"m4r_code\\Data\\labelled_tokenised_caverlee_bots.csv", 'r', encoding = "utf-8") as t:
        for i, line in enumerate(t):
            samples_bots.append(eval(line))
            if i == n:
                break
    """
    
    return df_humans, df_bots
    # return samples_humans, samples_bots

# Shuffling the data:
# And splitting into training and validation samples
def split_data(samples_humans, samples_bots):
    # Method 1:
    # Create pandas dataframe, shuffle using sample
    # data = pd.DataFrame(data={"tokenised_tweet": samples, "is_bot": labels})
    # data = data.sample(f=1)
    
    # Check to see if input is a pandas dataframe or a list:
    try:
        samples_humans = samples_humans.values
        samples_bots = samples_bots.values
    except:
        pass
    
    # Boolean variable determining if tweet is from a bot or not:
    no_of_humans = len(samples_humans)
    no_of_bots = len(samples_bots)
    labels = [0]*len(samples_humans) + [1]*len(samples_bots)
    samples = np.append(samples_humans, samples_bots)
    
    # Method 2:
    seed = 1337
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
    
    # Split into training and validation sets
    validation_split = 0.2
    num_validation_samples = int(validation_split * len(samples))
    train_samples = samples[:-num_validation_samples]
    val_samples = samples[-num_validation_samples:]
    train_labels = labels[:-num_validation_samples]
    val_labels = labels[-num_validation_samples:]
    
    
    return train_samples, val_samples, train_labels, val_labels




def load_embeddings(train_samples, max_tokens = 20000):
    """
    # Opens the glove.twitter.27B.25d.txt file
    # Converts to a dictionary
    # Removes any rows that may cause an error
    # (the files are sometimes a bit broken because they are the concatenation of
    #    lots of smaller files)
    """
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
    text_ds = tf.data.Dataset.from_tensor_slices(train_samples).batch(128)
    vectorizer.adapt(text_ds)
    
    return vectorizer
        



# Opening the GloVe embeddings file
def load_embeddings():
    """
    # Opens the glove.twitter.27B.25d.txt file
    # Converts to a dictionary
    # Removes any rows that may cause an error
    # (the files are sometimes a bit broken because they are the concatenation of
    #    lots of smaller files)
    """
    embeddings_dict = {}
    
    # Opening glove.twitter.27B.25d.txt
    with open(path_to_embeddings, 'r', encoding = "utf-8") as t:
        list_i = [] # list of lines ignored
        count_l = 0 # current line
        count_i = 0 # current no. of things ignored
        for line in t:
            count_l += 1
            sys.stdout.write("\rReading line %i" % count_l)
            # ----
            word, vector = line.split(maxsplit = 1)
            vector = np.fromstring(vector, dtype = "f", sep = " ")
            if count_l == 1:
                m = vector.shape[0]
            if vector.shape[0] != m:
                list_i.append(count_l)
                count_i += 1
            else:
                embeddings_dict[word] = vector
    
    print("\n", count_i, "lines ignored, at row(s):", str(list_i))

    return embeddings_dict    


# Creating an embedding matrix
def embedding_matrix(tokeniser_directory, embeddings_index, word_index, vocabulary_size=30000):
    """
    embeddings_index = embeddings_dict ()
    
    """
    


            
            
            
            
            
            