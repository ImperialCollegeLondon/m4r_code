"""
Simply just tokenising and pickling
"""


import sys
import numpy as np
import pickle
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser





def tokenising_caverlee():
    """
    caverlee_2011 contains:
    2353473 bot tweets and 3259693 human tweets
    --------
    This code:
    1. Loads the FULL caverlee datasets
    2. Iterates through the lines, extracts tweets and tokenises them
    3. Appends the tokenised tweets to samples list
    4. Creates a corresponding labels list
    5. Writes them to a pickle file
    --------
    This should only be run once! (To avoid overwriting the file!)
    --------
    Takes approximately 55 minutes to run
    """
    path_to_caverlee = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Indiana Dataset\\caverlee-2011\\social_honeypot_icwsm_2011\\"
    path_to_human_tweets = path_to_caverlee + "legitimate_users_tweets.txt"
    path_to_bot_tweets = path_to_caverlee + "content_polluters_tweets.txt"
    path_to_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data\\"

    samples_humans = []
    samples_bots = []
    labels = []

    with open(path_to_human_tweets,'r', encoding = "utf-8") as r:
        for i, line in enumerate(r):
            sys.stdout.write("\rReading line %i, %s%% done" % (i+1, 100*round(i/3259693, 4)))
            tweet = line.split("\t")[2:-1]
            tweet = " ".join(tweet)
            tokens = text_tokeniser(tweet)
            s = " ".join(tokens)
            samples_humans.append(s)
            
    print("\nFinished tokenising human tweets\n...")

    with open(path_to_bot_tweets, 'r', encoding = "utf-8") as r:
        for i, line in enumerate(r):
            sys.stdout.write("\rReading line %i, %s%% done" % (i+1, 100*round(i/2353473, 4)))
            tweet = line.split("\t")[2:-1]
            tweet = " ".join(tweet)
            tokens = text_tokeniser(tweet)
            s = " ".join(tokens)
            samples_bots.append(s)
            
    print("Pickling...")

    samples = samples_humans + samples_bots
    labels = [0]*len(samples_humans) + [1] * len(samples_bots)
    pickle.dump(samples, open(path_to_data + "caverlee_2011_samples.p", "wb"))
    pickle.dump(labels, open(path_to_data + "caverlee_2011_labels.p", "wb"))
    print("Done.")
    # wb = write binary
