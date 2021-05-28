"""
Created on Wed Feb 17 07:39:53 2021

@author: fangr
"""

# *- MODULES -*
import sys, pickle
import pandas as pd
import numpy as np
import datetime
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser
# import tensorflow as tf
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# *- FILE PATHS -*
indiana = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Indiana Dataset\\"
italy   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Cresci Italian Dataset\\"
dset = ["Astroturf", "Cresci RTbust 2019", "Gilani 2017", "Midterm 2018", "Political Bots 2019", "Varol 2017"]
tset = ["collected_astroturf_tweets.p", "collected_cresci-rtbust-2019_tweets.p", "collected_gilani-2017_tweets_reduced_columns.p", "collected_midterm_2018_tweets.p", "collected_political-bots-2019_tweets.p", "collected_varol-2017_tweets.p"]
lset = ["astroturf.tsv", "cresci-rtbust-2019_labels.tsv", "gilani-2017_labels.tsv", "midterm-2018_labels.tsv", "political-bots-2019.tsv", "varol-2017.tsv"]
bset = ["pandas_social_spambots_1_tweets.p", "pandas_social_spambots_2_tweets.p", "pandas_social_spambots_3_tweets.p", "pandas_traditional_spambots_1_tweets.p"]
italset = ["social_spambots_1", "social_spambots_2", "social_spambots_3", "traditional_spambots_1"]


# *- Importing Datasets -*
def print_info():
    for i in range(6):
        print(dset[i], "dataset:")
        df = pickle.load(open(indiana + tset[i], "rb"))
        lb = pd.read_csv(indiana + lset[i], delim_whitespace = True, header = None); lb.columns = ["user.id", "class"]
        df = pd.merge(df, lb, on = ["user.id"])
        print("# Tweets =", len(df)); print("# EN Tweets =", len(df[df["lang"] == "en"])); print("# 'non repeated' =", len(set(df["full_text"])))
        print("# Users =", len(set(df["user.id"])))
        if i == 5:
            print("# Human tweets:", len(df[df["class"] == 0]))
        else:
            print("# Human tweets:", len(df[df["class"] == "human"]))
        print("===============================")


def create_bot_tweet_dataset_indiana(i = 0):
    """
    Training data (class 1 := known bots)
    Number of samples total: ~250,000
    """
    # num_collect = [5000	, 15000, 50000,	1000,	500,	15000,	25000,	80000,	15000,	25000] # old one, for new one, added '1500 + midterm, 5000 + astroturf, 8000 + varol, 4000 + gilani'
    num_collect = [5000	+ 5000, 15000, 50000 + 4000,	1000 + 1500,	500,	15000 + 8000,	25000,	80000,	15000,	25000]
    tweet_columns = ["tweet", "num_hashtags", "num_mentions", "num_urls", 'retweet_count', 'favorite_count', "len_tweet"]
    user_columns_count = ["user.age", "user.followers_count", "user.friends_count", "user.statuses_count", "user.favourites_count", "user.listed_count"]
    user_columns_bool = ["user.default_profile", "user.default_profile_image", "user.geo_enabled", "user.verified"] # , "user.has_location"
    derived_columns = ["user.len_name", "user.len_screen_name", "user.num_digits_name", "user.num_digits_screen_name", "user.len_description"]
    class_columns = ["full_text", "id", "user.id", "label", "dataset"]
    relevant_columns = tweet_columns + user_columns_count + user_columns_bool + derived_columns + class_columns
    savenames = ["bot_sample_astroturf.p", "bot_sample_cresci-rtbust-2019.p", "bot_sample_gilani-2017.p", "bot_sample_midterm_2018.p", "bot_sample_political-bots-2019.p", "bot_sample_varol-2017.p"]
    # Collecting Tweets from the Indiana Dataset
    # bot_dataset = pd.DataFrame()
    
    sys.stdout.write("\rReading "+str(i))
    df = pickle.load(open(indiana + tset[i], "rb"))
    lb = pd.read_csv(indiana + lset[i], delim_whitespace = True, header = None); lb.columns = ["user.id", "class"]
    df = pd.merge(df, lb, on = ["user.id"])
    if i != 5:
        df["label"] = [0 if x == "human" else 1 for x in df["class"]]
    else:
        df["label"] = df["class"]
    # remove humans...
    df = df[df["label"] == 1]
    
    # derived features...
    # ===================
    # METADATA
    # ===================
    df["num_hashtags"] = df["entities.hashtags"].str.len()
    df["num_urls"] = df["entities.urls"].str.len()
    df["num_mentions"] = df["entities.user_mentions"].str.len()
    # ===================
    # AGE: Calculating the age of the account
    # ===================
    # pd.to_datetime(df["user.created_at"], format='%a %b %d %H:%M:%S %z %Y').dt.tz_localize(None)
    harvest_date = datetime.datetime(2021,2,15,0,0,0)
    df["user.age"] = (harvest_date - pd.to_datetime(df["user.created_at"], format='%a %b %d %H:%M:%S %z %Y').dt.tz_localize(None)).dt.days
    # date that I "harvested" the tweets/account data: 15th February 2021 @ 00:00:00 @ UTC +0000
    # ===================
    # DIGITS IN SCREENNAME ETC...
    # ===================
    df["user.len_screen_name"] = df["user.screen_name"].str.len()
    df["user.len_name"] = df["user.name"].str.len()
    df["user.num_digits_screen_name"] = df["user.screen_name"].str.count(r"\d")
    df["user.num_digits_name"] = df["user.name"].str.count(r"\d")
    # df["user.has_location"] = df["user.location"].notna() # <---- this is stupid because nearly all of them have locations
    # ===================
    # LENGTH OF TWEET AND DESCRIPTION
    # ===================
    df["len_tweet"] = df["full_text"].str.len()
    df["user.len_description"] = df["user.description"].str.len()
    df["dataset"] = dset[i]
    # ===================
    # TOKENISING
    # ===================
    df["tweet"] = df["full_text"].apply(lambda x : text_tokeniser(x))
    
    pickle.dump(df[relevant_columns].sample(num_collect[i], random_state = i * 1349565), open(indiana + savenames[i],"wb"))
    df = None
        #bot_dataset = bot_dataset.append(df[relevant_columns].sample(num_collect[i]), ignore_index = True)
    #return df[relevant_columns].sample(num_collect[i], random_state = i * 1349565)

    
def create_bot_tweet_dataset_italy(i = 0):
    num_collect = [25000,	80000,	15000,	25000]
    tweet_columns = ["tweet", "num_hashtags", "num_mentions", "num_urls", 'retweet_count', 'favorite_count', "len_tweet"]
    user_columns_count = ["user.age", "user.followers_count", "user.friends_count", "user.statuses_count", "user.favourites_count", "user.listed_count"]
    user_columns_bool = ["user.default_profile", "user.default_profile_image", "user.geo_enabled", "user.verified"] # , "user.has_location"
    derived_columns = ["user.len_name", "user.len_screen_name", "user.num_digits_name", "user.num_digits_screen_name", "user.len_description"]
    class_columns = ["full_text", "id", "user.id", "label", "dataset"]
    relevant_columns = tweet_columns + user_columns_count + user_columns_bool + derived_columns + class_columns

    df  = pickle.load(open(italy + "pandas_" + italset[i] + "_tweets.p", "rb"))
    dfu = pd.read_csv(italy + italset[i] + "_users.csv")

    # sampling
    df = df.sample(num_collect[i], random_state = (i+6)*1349565)
    
    if i == 3:
        dfu["created_at"] = dfu["timestamp"]
        dfu["age"] = (pd.to_datetime(dfu["crawled_at"], format = "%Y-%m-%d %H:%M:%S") - pd.to_datetime(dfu["created_at"], format='%Y-%m-%d %H:%M:%S')).dt.days
    
    # derived features...
    # ===================
    # AGE: Calculating the age of the account
    # ===================
    else:
        dfu["age"] = (pd.to_datetime(dfu["crawled_at"], format = "%Y-%m-%d %H:%M:%S") - pd.to_datetime(dfu["created_at"], format='%a %b %d %H:%M:%S %z %Y').dt.tz_localize(None)).dt.days
    # ===================
    # DIGITS IN SCREENNAME ETC...
    # ===================
    dfu["user.len_screen_name"] = dfu["screen_name"].str.len()
    dfu["user.len_name"] = dfu["name"].str.len()
    dfu["user.num_digits_screen_name"] = dfu["screen_name"].str.count(r"\d")
    dfu["user.num_digits_name"] = dfu["name"].str.count(r"\d")
    # ===================
    # LENGTH OF TWEET AND DESCRIPTION
    # ===================
    df["len_tweet"] = df["text"].str.len()
    dfu["user.len_description"] = dfu["description"].str.len()
    df["dataset"] = italset[i]
    df["label"] = 1.0
    # ===================
    # TOKENISING
    # ===================
    df["tweet"] = df["text"].apply(lambda x : text_tokeniser(x))
    
    
    # Changing Column names...
    df.rename(columns={
        'tweet_id': 'id',
        "user_id" : "user.id",
        'text': 'full_text'},
        inplace=True)
    dfurenamedict  = {"id": "user.id", **{x[5:] : x for x in user_columns_count}, **{x[5:] : x for x in user_columns_bool}}
    dfu.rename(columns=dfurenamedict, inplace=True)
    
    # Merging...
    df = pd.merge(df, dfu, on = ["user.id"])
    pickle.dump(df[relevant_columns], open(italy + "bot_sample_" + italset[i]+".p","wb"))
    df = None
    
def create_human_tweet_dataset():
    tweet_columns = ["tweet", "num_hashtags", "num_mentions", "num_urls", 'retweet_count', 'favorite_count', "len_tweet"]
    user_columns_count = ["user.age", "user.followers_count", "user.friends_count", "user.statuses_count", "user.favourites_count", "user.listed_count"]
    user_columns_bool = ["user.default_profile", "user.default_profile_image", "user.geo_enabled", "user.verified"] # , "user.has_location"
    derived_columns = ["user.len_name", "user.len_screen_name", "user.num_digits_name", "user.num_digits_screen_name", "user.len_description"]
    class_columns = ["full_text", "id", "user.id", "label", "dataset"]
    relevant_columns = tweet_columns + user_columns_count + user_columns_bool + derived_columns + class_columns
    
    df = pickle.load(open(italy + "pandas_3200000_stratified_shuffled.p", "rb"))
    
    # removing bots
    df = df[df["bot"] == 0]
    # removing entries with missing tweets
    # remove entries missing tweets...
    df = df[df["text"].str.len() != 0]
    
    # ===================
    # SAMPLING
    # ===================
    df = df.sample(200000, random_state = 7*1349565)
    
    dfu = pd.read_csv(italy + "genuine_accounts_users.csv")
    
    # ===================
    # AGE: Calculating the age of the account
    # ===================
    dfu["age"] = (pd.to_datetime(dfu["crawled_at"], format = "%Y-%m-%d %H:%M:%S") - pd.to_datetime(dfu["created_at"], format='%a %b %d %H:%M:%S %z %Y').dt.tz_localize(None)).dt.days
    
    
    # ===================
    # CHANGE COLUMN NAMES
    # ===================
    df.rename(columns={
        'tweet_id': 'id',
        "user_id" : "user.id",
        'text': 'full_text'},
        inplace=True)
    dfurenamedict  = {"id": "user.id", **{x[5:] : x for x in user_columns_count}, **{x[5:] : x for x in user_columns_bool}}
    dfu.rename(columns=dfurenamedict, inplace=True)
    
    # ===================
    # DIGITS IN SCREENNAME ETC...
    # ===================
    dfu["user.len_screen_name"] = dfu["screen_name"].str.len()
    dfu["user.len_name"] = dfu["name"].str.len()
    dfu["user.num_digits_screen_name"] = dfu["screen_name"].str.count(r"\d")
    dfu["user.num_digits_name"] = dfu["name"].str.count(r"\d")
    # ===================
    # LENGTH OF TWEET AND DESCRIPTION
    # ===================
    df["len_tweet"] = df["full_text"].str.len()
    dfu["user.len_description"] = dfu["description"].str.len()
    df["dataset"] = "Cresci 2017"
    # ===================
    # TOKENISING
    # ===================
    df["tweet"] = df["full_text"].apply(lambda x : text_tokeniser(x))
    df["label"] = 0.0
    
    df = pd.merge(df, dfu, on = ["user.id"])
    pickle.dump(df[relevant_columns], open(italy + "human_sample_genuine_accounts.p","wb"))
    
    return df


def create_human_tweet_dataset_gilani():
    """
    Training data (class 1 := known bots)
    Number of samples total: ~250,000
    """
    num_collect = [5000	, 15000, 50000,	1000,	500,	15000,	25000,	80000,	15000,	25000]
    tweet_columns = ["tweet", "num_hashtags", "num_mentions", "num_urls", 'retweet_count', 'favorite_count', "len_tweet"]
    user_columns_count = ["user.age", "user.followers_count", "user.friends_count", "user.statuses_count", "user.favourites_count", "user.listed_count"]
    user_columns_bool = ["user.default_profile", "user.default_profile_image", "user.geo_enabled", "user.verified"] # , "user.has_location"
    derived_columns = ["user.len_name", "user.len_screen_name", "user.num_digits_name", "user.num_digits_screen_name", "user.len_description"]
    class_columns = ["full_text", "id", "user.id", "label", "dataset"]
    relevant_columns = tweet_columns + user_columns_count + user_columns_bool + derived_columns + class_columns
    savename = "human_sample_gilani-2017.p"
    # Collecting Tweets from the Indiana Dataset
    # bot_dataset = pd.DataFrame()
    
    df = pickle.load(open(indiana + "collected_gilani-2017_tweets_reduced_columns.p", "rb"))
    lb = pd.read_csv(indiana + "gilani-2017_labels.tsv", delim_whitespace = True, header = None); lb.columns = ["user.id", "class"]
    df = pd.merge(df, lb, on = ["user.id"])
    df["label"] = [0 if x == "human" else 1 for x in df["class"]]
    
    # remove bots...
    df = df[df["label"] == 0]
    
    # derived features...
    # ===================
    # METADATA
    # ===================
    df["num_hashtags"] = df["entities.hashtags"].str.len()
    df["num_urls"] = df["entities.urls"].str.len()
    df["num_mentions"] = df["entities.user_mentions"].str.len()
    # ===================
    # AGE: Calculating the age of the account
    # ===================
    # pd.to_datetime(df["user.created_at"], format='%a %b %d %H:%M:%S %z %Y').dt.tz_localize(None)
    harvest_date = datetime.datetime(2021,2,15,0,0,0)
    df["user.age"] = (harvest_date - pd.to_datetime(df["user.created_at"], format='%a %b %d %H:%M:%S %z %Y').dt.tz_localize(None)).dt.days
    # date that I "harvested" the tweets/account data: 15th February 2021 @ 00:00:00 @ UTC +0000
    # ===================
    # DIGITS IN SCREENNAME ETC...
    # ===================
    df["user.len_screen_name"] = df["user.screen_name"].str.len()
    df["user.len_name"] = df["user.name"].str.len()
    df["user.num_digits_screen_name"] = df["user.screen_name"].str.count(r"\d")
    df["user.num_digits_name"] = df["user.name"].str.count(r"\d")
    # df["user.has_location"] = df["user.location"].notna() # <---- this is stupid because nearly all of them have locations
    # ===================
    # LENGTH OF TWEET AND DESCRIPTION
    # ===================
    df["len_tweet"] = df["full_text"].str.len()
    df["user.len_description"] = df["user.description"].str.len()
    df["dataset"] = "Gilani 2017"
    # ===================
    # TOKENISING
    # ===================
    df["tweet"] = df["full_text"].apply(lambda x : text_tokeniser(x))
    
    pickle.dump(df[relevant_columns].sample(50000, random_state = 8 * 1349565), open(indiana + savename,"wb"))
    return df
