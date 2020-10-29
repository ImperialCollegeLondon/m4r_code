"""
Preparing and processing labelled datasets and saving them to the 'Data' folder
of m4r_code repository
"""


# Modules to import
import json
import pandas as pd
import sys
import numpy as np
import csv
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser
import pickle


# File paths
path_to_m4r = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\"
path_to_m4r_code = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\"


# Path to caverlee dataset folder
path_to_caverlee = path_to_m4r + "Indiana Dataset\\caverlee-2011\\social_honeypot_icwsm_2011\\"

# Path to GloVe embeddings (Twitter corpus)
path_to_embeddings = path_to_m4r + "GloVe\\glove.twitter.27B\\glove.twitter.27B.25d.txt"



# Opening the labelled caverlee-2011 dataset from the Indiana dataset
# Then tokenising it using full_text_tokeniser
# saving it to a folder in m4r-code
def tokenising_caverlee_data_1(n = None):
    """
    # Opens the caverlee-2011 dataset from the Indiana dataset
    # Loads tweets from legitimate users and content polluters
    # Saves them to the folder 'Data' in m4r-code
    # ONLY COLLECTS TWEETS, NOT METADATA
    n = number of tweets to collect from each data source
        (int, or None if you want to collect all of them)
    """
    print("Begin")
    
    
    samples_human = []
    samples_bot = []
    
    # Opening the legitimate users tweets and the folder to write to
    with open(path_to_caverlee+"legitimate_users_tweets.txt", 'r', encoding = "utf-8") as a, open(path_to_m4r_code+"Data\\labelled_tokenised_caverlee_humans.csv", 'w', encoding = "utf-8", newline='') as b:
        writer = csv.writer(b)
        for i, line in enumerate(a):
            sys.stdout.write("\rReading line %i" % i)
            values = line.split("\t")
            tweet = "".join(values[2:-1])
            tokenised_tweet = [text_tokeniser(tweet)]
            writer.writerow(tokenised_tweet)
            if i+1 == n:
                break
    
    # Signifying that the first batch of data has been written
    print("\nLegitimate Users Done \n...")
    
    # Opening the content polluters tweets and the folder to write to
    with open(path_to_caverlee+"content_polluters_tweets.txt", 'r', encoding = "utf-8") as a, open(path_to_m4r_code+"Data\\labelled_tokenised_caverlee_bots.csv", 'w', encoding = "utf-8", newline='') as b:
        writer = csv.writer(b)
        for i, line in enumerate(a):
            sys.stdout.write("\rReading line %i" % i)
            values = line.split("\t")
            tweet = "".join(values[2:-1])
            tokenised_tweet = [text_tokeniser(tweet)]
            writer.writerow(tokenised_tweet)
            if i+1 == n:
                break

    # Signifying that the first batch of data has been written
    print("\nContent Polluters Done")
    print("Done")



# Loading the tokenised tweets, converting it to a numpy array:
# creating a samples and labels np array
# Then saving it to a pickle file

def pickling_caverlee_data_1(n = None):
    
    samples_humans = np.array([])
    samples_bots = np.array([])
    
    open(path_to_m4r_code+"Data\\labelled_tokenised_caverlee_humans_nparray.p", 'wb') as b
    
    # Opening the legitimate users tweets and the folder to write to
    with open(path_to_caverlee+"legitimate_users_tweets.txt", 'r', encoding = "utf-8") as a:
        for i, line in enumerate(a):
            sys.stdout.write("\rReading line %i" % i)
            values = line.split("\t")
            tweet = "".join(values[2:-1])
            tokenised_tweet = np.array(text_tokeniser(tweet))
            samples_humans = samples_humans.append
        
        
        writer = csv.writer(b)
        for i, line in enumerate(a):
            sys.stdout.write("\rReading line %i" % i)
            values = line.split("\t")
            tweet = "".join(values[2:-1])
            tokenised_tweet = [text_tokeniser(tweet)]
            writer.writerow(tokenised_tweet)
            if i+1 == n:
                break
    
    # Signifying that the first batch of data has been written
    print("\nLegitimate Users Done \n...")
    
    # Opening the content polluters tweets and the folder to write to
    with open(path_to_caverlee+"content_polluters_tweets.txt", 'r', encoding = "utf-8") as a, open(path_to_m4r_code+"Data\\labelled_tokenised_caverlee_bots.csv", 'w', encoding = "utf-8", newline='') as b:
        writer = csv.writer(b)
        for i, line in enumerate(a):
            sys.stdout.write("\rReading line %i" % i)
            values = line.split("\t")
            tweet = "".join(values[2:-1])
            tokenised_tweet = [text_tokeniser(tweet)]
            writer.writerow(tokenised_tweet)
            if i+1 == n:
                break

    # Signifying that the first batch of data has been written
    print("\nContent Polluters Done")
    print("Done")
    
    

