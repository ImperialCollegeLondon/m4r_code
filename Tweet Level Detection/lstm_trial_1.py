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



# Opening the labelled caverlee-2011 dataset from the Indiana dataset
def load_data(n = None, save = False):
    """
    # Opens the caverlee-2011 dataset from the Indiana dataset
    # Loads tweets from legitimate users and content polluters
    # Shuffles them and splits them into a training and a testing dataset
    n = number of tweets to collect from each data source
        (int, or None if you want to collect all of them)
    save = whether to save or output as a file
    """
    
    
    samples = []
    # Opening the legitimate users tweets:
    with open(path_to_caverlee+"legitimate_users_tweets.txt", 'r', encoding = "utf-8") as t:
        count = 0
        for i, line in enumerate(t):
            count += 1
            sys.stdout.write("\rReading line %i" % i+1)
            values = line.split("\t")
            tweet = "".join(values[2:-1])
            tokenised_tweet = text_tokeniser(tweet)
            samples.append(tokenised_tweet)
            if count == n:
                break
    no_of_legitimate = len(samples)
    # Opening the content polluters tweets:
    print("\nMoving onto content polluters tweets\n...")
    with open(path_to_caverlee+"content_polluters_tweets.txt", 'r', encoding = "utf-8") as t:
        count = 0
        for i, line in enumerate(t):
            count += 1
            sys.stdout.write("\rReading line %i" % i+1)
            values = line.split("\t")
            tweet = "".join(values[2:-1])
            tokenised_tweet = text_tokeniser(tweet)
            samples.append(tokenised_tweet)
            if count == n:
                break
    no_of_polluters = len(samples) - no_of_legitimate
    
    # Boolean variable determining if tweet is from a bot or not:
    labels = [0]*no_of_legitimate + [1]*no_of_polluters
    
    
    # Shuffling the data:
    
    # Method 1:
    # Create pandas dataframe, shuffle using sample
    # data = pd.DataFrame(data={"tokenised_tweet": samples, "is_bot": labels})
    # data = data.sample(f=1)
    
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


    

def load_embeddings(max_tokens = 20000):
    """
    # Opens the glove.twitter.27B.25d.txt file
    # Converts to a dictionary
    # Removes any rows that may cause an error
    # (the files are sometimes a bit broken because they are the concatenation of
    #    lots of smaller files)
    """
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
    embeddings_dict = {}
    



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
    


            
            
            
            
            
            