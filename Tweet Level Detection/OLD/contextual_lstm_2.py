"""
Created on Wed Feb 17 07:39:53 2021

@author: fangr
"""

# *- MODULES -*
import sys, pickle
import pandas as pd
import numpy as np
# import tensorflow as tf
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# *- FILE PATHS -*
indiana = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Indiana Dataset\\"
italy   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Cresci Italian Dataset\\"
dset = ["Astroturf", "Cresci RTbust 2019", "Gilani 2017", "Midterm 2018", "Political Bots 2019", "Varol 2017"]
tset = ["collected_astroturf_tweets.p", "collected_cresci-rtbust-2019_tweets.p", "collected_gilani-2017_tweets.p", "collected_midterm_2018_tweets.p", "collected_political-bots-2019_tweets.p", "collected_varol-2017_tweets.p"]
lset = ["astroturf.tsv", "cresci-rtbust-2019_labels.tsv", "gilani-2017_labels.tsv", "midterm-2018_labels.tsv", "political-bots-2019.tsv", "varol-2017.tsv"]
bset = ["pandas_social_spambots_1_tweets.p", "pandas_social_spambots_2_tweets.p", "pandas_social_spambots_3_tweets.p", "pandas_traditional_spambots_1_tweets.p"]


# *- Importing Datasets -*
def print_info():
    for i in range(5, 6):
        print(dset[i], "dataset:")
        df = pickle.load(open(indiana + tset[i], "rb"))
        lb = pd.read_csv(indiana + lset[i], delim_whitespace = True, header = None); lb.columns = ["user.id", "class"]
        df = pd.merge(df, lb, on = ["user.id"])
        print("# Tweets =", len(df)); print("# EN Tweets =", len(df[df["lang"] == "en"])); print("# 'non repeated' =", len(set(df["full_text"])))
        print("# Users =", len(set(df["user.id"])))
        print("===============================")

def create_bot_tweet_dataset():
    """
    Training data (class 1 := known bots)
    """
    
    

    
    









