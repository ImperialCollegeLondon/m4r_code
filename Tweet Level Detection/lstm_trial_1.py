# -*- coding: utf-8 -*-
"""

"""

# Modules to import
import json
import pandas as pd

# File paths
path_to_m4r = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\"
path_to_caverlee = path_to_m4r + "Indiana Dataset\\caverlee-2011\\social_honeypot_icwsm_2011\\"



# Opening the labelled caverlee-2011 dataset from the Indiana dataset
def load_data():
    """
    # Opens the caverlee-2011 dataset from the Indiana dataset
    # Loads tweets from legitimate users and content polluters
    # Shuffles them and splits them into a training and a testing dataset
    """
    
    # Opening the legitimate_users_tweets.txt 
    legitimate_tweets_data = pd.read_csv(path_to_caverlee+"legitimate_users_tweets.txt", delimiter="\t", header = None, usecols = [2])
    
    
    
        
    return legitimate_tweets_data
