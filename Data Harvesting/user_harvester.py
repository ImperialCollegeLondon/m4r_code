"""
Title:
User Harvester

What it does:
Collects tweets based on user ID's

Parameters to change before running:
overwrite:
    True => overwrites the file 
    False => appends the data
new_header:
    True => creates a new header row (automatically True if overwriting)
    False => doesn't create a new header row (leave False if overwrite = False)
n:
    maximum number of tweets to collect upon running the code
    set to None if you want to max out at the Twitter rate limit
"""
# *- OVERWRITE? -*
overwrite = False

# *- INITIALISE HEADER? -*
new_header = False

# *- NUMBER OF TWEETS TO COLLECT? -*
n = 200

# *- PACKAGES -*
import tweepy
import sys, os
from datetime import date
import csv
import time
import pandas as pd
import pickle
import json



# *- FILE PATHS -*
# current folder we're in:
folder_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting"
# file to write to:
file_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Collected Tweets\\cresci_rtbust_2019.csv"
# Adding current directory so that we can access the Twitter API keys
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
# Indiana dataset...
path_to_indiana = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Indiana Dataset\\"

# *- TWITTER API KEYS -*
try:
    import config
    consumer_key = config.consumer_key
    consumer_secret = config.consumer_secret
    access_token = config.access_token
    access_token_secret = config.access_token_secret
except:
    print("Authentication information is missing")
    
# *- ACCESSING TWITTER API -*
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# *- COLLECTING USERS FROM REPLY NETWORK THAT DON'T HAVE ACCOUNT DATA -* #
def collect_reply_network():
    users_all = None
    # SEE M4R_REPOSITORY\\OTHER\\10_APRIL_2021.PY #



# *- COLLECTING CRESCI_RTBUST -*
def collect_cresci_rtbust():
    users_cresci_rtbust = pd.read_csv(path_to_indiana + "cresci-rtbust-2019_labels.tsv", sep = "\t", header = None)
    users_cresci_rtbust.columns = ["user_id", "class"]
    user_ids = users_cresci_rtbust["user_id"].to_numpy()
    # user_ids = ["390617262"] # bot account, test of just one
    # user_ids = [390617262, 8972349871603847]
    
    
    collected_user_tweets = pd.DataFrame()
    skip_ids = []
    
    for i, u in enumerate(user_ids):
        try:
            tweets = api.user_timeline(user_id = u, 
                                       # 200 is the maximum allowed count
                                       count = 200,
                                       include_rts = True,
                                       # Necessary to keep full_text 
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode = 'extended',
                                       wait_on_rate_limit = True,
                                       wait_on_rate_limit_notify = True
                                       )
            json_data = [r._json for r in tweets]
            dfj = pd.json_normalize(json_data)
            collected_user_tweets = collected_user_tweets.append(dfj, ignore_index = True)
            sys.stdout.write("\rReading " + str(i))
        except:
            skip_ids.append(u)
            sys.stdout.write("\rReading " + str(i))
            
    pickle.dump(collected_user_tweets, open( path_to_indiana + "collected_cresci-rtbust-2019_tweets.p" , "wb"))
    print("Done")
    return collected_user_tweets

# *- COLLECTING MIDTERM 2018 -*
def collect_midterm_2018():
    """
    This is a HUGE dataset... there are 50538 total accounts
    - the FIRST 42446 users are bots
    - the LAST 8092 users are human
    So, we'll collect them in batches... of 5,000?
    ==========================================================================
    Out of the 42,446 bot users, only 44 of them still are 'active',
    these are stored in the file valid_midterm_2018_bot_ids.p
    """
    users_midterm_2018 = pd.read_csv(path_to_indiana + "midterm-2018_labels.tsv", sep = "\t", header = None)
    users_midterm_2018.columns = ["user_id", "class"]
    # filtering out human users
    users_midterm_2018 = users_midterm_2018[users_midterm_2018["class"] == "bot"]
    # shuffling the remaining bot users (in a set random seed state)
    users_midterm_2018 = users_midterm_2018.sample(frac = 1.0, random_state = 101)
    human_user_ids = users_midterm_2018["user_id"].to_numpy()
    
    user_ids = pickle.load(open(path_to_indiana  + "valid_midterm_2018_bot_ids.p", "rb"))
    
    
    collected_user_tweets = pd.DataFrame()
    skip_ids = []
    unskipped_ids = []
    
    for i, u in enumerate(user_ids):
        try:
            tweets = api.user_timeline(user_id = u, 
                                       # 200 is the maximum allowed count
                                       count = 200,
                                       include_rts = True,
                                       # Necessary to keep full_text 
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode = 'extended',
                                       wait_on_rate_limit = True,
                                       wait_on_rate_limit_notify = True
                                       )
            json_data = [r._json for r in tweets]
            dfj = pd.json_normalize(json_data)
            collected_user_tweets = collected_user_tweets.append(dfj, ignore_index = True)
            sys.stdout.write("\rReading " + str(i))
            unskipped_ids.append(u)
        except:
            skip_ids.append(u)
            sys.stdout.write("\rReading " + str(i))
            
    pickle.dump(collected_user_tweets, open( path_to_indiana + "collected_midterm_2018_tweets.p" , "wb"))
    print("Done")
    print(len(skip_ids), "skipped")
    return collected_user_tweets #, skip_ids, unskipped_ids

# *- COLLECTING GILANI 2O17 -*
def collect_gilani():
    users_gilani = pd.read_csv(path_to_indiana + "gilani-2017_labels.tsv", sep = "\t", header = None)
    users_gilani.columns = ["user_id", "class"]
    user_ids = users_gilani["user_id"].to_numpy()
    # user_ids = ["390617262"] # bot account, test of just one
    # user_ids = [390617262, 8972349871603847]
    
    
    collected_user_tweets = pd.DataFrame()
    skip_ids = []
    
    for i, u in enumerate(user_ids):
        try:
            tweets = api.user_timeline(user_id = u, 
                                       # 200 is the maximum allowed count
                                       count = 200,
                                       include_rts = True,
                                       # Necessary to keep full_text 
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode = 'extended',
                                       wait_on_rate_limit = True,
                                       wait_on_rate_limit_notify = True
                                       )
            json_data = [r._json for r in tweets]
            dfj = pd.json_normalize(json_data)
            collected_user_tweets = collected_user_tweets.append(dfj, ignore_index = True)
            sys.stdout.write("\rReading " + str(i))
        except:
            skip_ids.append(u)
            sys.stdout.write("\rReading " + str(i))
            
    pickle.dump(collected_user_tweets, open( path_to_indiana + "collected_gilani-2017_tweets.p" , "wb"))
    print("Done")
    print(len(skip_ids), "skipped, out of", len(user_ids), "total users")
    return collected_user_tweets


# *- COLLECTING POLITICAL BOTS 2019 -*
def collect_political_bots_2019():
    users_political = pd.read_csv(path_to_indiana + "political-bots-2019.tsv", sep = "\t", header = None)
    users_political.columns = ["user_id", "class"]
    user_ids = users_political["user_id"].to_numpy()
    # user_ids = ["390617262"] # bot account, test of just one
    # user_ids = [390617262, 8972349871603847]
    
    
    collected_user_tweets = pd.DataFrame()
    skip_ids = []
    useable_ids = []
    
    for i, u in enumerate(user_ids):
        try:
            tweets = api.user_timeline(user_id = u, 
                                       # 200 is the maximum allowed count
                                       count = 200,
                                       include_rts = True,
                                       # Necessary to keep full_text 
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode = 'extended',
                                       wait_on_rate_limit = True,
                                       wait_on_rate_limit_notify = True
                                       )
            json_data = [r._json for r in tweets]
            dfj = pd.json_normalize(json_data)
            collected_user_tweets = collected_user_tweets.append(dfj, ignore_index = True)
            sys.stdout.write("\rReading " + str(i))
            useable_ids.append(u)
        except:
            skip_ids.append(u)
            sys.stdout.write("\rReading " + str(i))
            
    pickle.dump(collected_user_tweets, open( path_to_indiana + "collected_political-bots-2019_tweets.p" , "wb"))
    print("Done")
    print(len(skip_ids), "skipped, out of", len(user_ids), "total users")
    return collected_user_tweets, useable_ids


# *- COLLECTING ASTROTURF 2020 -*
def collect_astroturf_2020():
    users_astroturf = pd.read_csv(path_to_indiana + "astroturf.tsv", sep = "\t", header = None)
    users_astroturf.columns = ["user_id", "class"]
    user_ids = users_astroturf["user_id"].to_numpy()
    # user_ids = ["390617262"] # bot account, test of just one
    # user_ids = [390617262, 8972349871603847]
    
    
    collected_user_tweets = pd.DataFrame()
    skip_ids = []
    useable_ids = []
    
    for i, u in enumerate(user_ids):
        try:
            tweets = api.user_timeline(user_id = u, 
                                       # 200 is the maximum allowed count
                                       count = 200,
                                       include_rts = True,
                                       # Necessary to keep full_text 
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode = 'extended',
                                       wait_on_rate_limit = True,
                                       wait_on_rate_limit_notify = True
                                       )
            json_data = [r._json for r in tweets]
            dfj = pd.json_normalize(json_data)
            collected_user_tweets = collected_user_tweets.append(dfj, ignore_index = True)
            sys.stdout.write("\rReading " + str(i))
            useable_ids.append(u)
        except:
            skip_ids.append(u)
            sys.stdout.write("\rReading " + str(i))
            
    pickle.dump(collected_user_tweets, open( path_to_indiana + "collected_astroturf_tweets.p" , "wb"))
    print("Done")
    print(len(skip_ids), "skipped, out of", len(user_ids), "total users")
    return collected_user_tweets, useable_ids

# *- COLLECTING VAROL 2017 -*
def collect_varol_2017():
    users_varol = pd.read_csv(path_to_indiana + "varol-2017.tsv", sep = "\t", header = None)
    users_varol.columns = ["user_id", "class"]
    user_ids = users_varol["user_id"].to_numpy()
    # user_ids = ["390617262"] # bot account, test of just one
    # user_ids = [390617262, 8972349871603847]
    
    
    collected_user_tweets = pd.DataFrame()
    skip_ids = []
    useable_ids = []
    
    for i, u in enumerate(user_ids):
        try:
            tweets = api.user_timeline(user_id = u, 
                                       # 200 is the maximum allowed count
                                       count = 200,
                                       include_rts = True,
                                       # Necessary to keep full_text 
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode = 'extended',
                                       wait_on_rate_limit = True,
                                       wait_on_rate_limit_notify = True
                                       )
            json_data = [r._json for r in tweets]
            dfj = pd.json_normalize(json_data)
            collected_user_tweets = collected_user_tweets.append(dfj, ignore_index = True)
            sys.stdout.write("\rReading " + str(i))
            useable_ids.append(u)
        except:
            skip_ids.append(u)
            sys.stdout.write("\rReading " + str(i))
            
    pickle.dump(collected_user_tweets, open( path_to_indiana + "collected_varol-2017_tweets.p" , "wb"))
    print("Done")
    print(len(skip_ids), "skipped, out of", len(user_ids), "total users")
    return collected_user_tweets, useable_ids

# *- COLLECTING VAROL 2017 -*
def collect_varol_2017_human():
    users_varol = pd.read_csv(path_to_indiana + "varol-2017.tsv", delim_whitespace = True, header = None)
    users_varol.columns = ["user_id", "class"]
    # filtering to only contain the human users
    users_varol = users_varol[users_varol["class"] == 0]
    user_ids = users_varol["user_id"].to_numpy()
    # user_ids = ["390617262"] # bot account, test of just one
    # user_ids = [390617262, 8972349871603847]
    
    
    collected_user_tweets = pd.DataFrame()
    skip_ids = []
    useable_ids = []
    
    for i, u in enumerate(user_ids):
        try:
            tweets = api.user_timeline(user_id = u, 
                                       # 200 is the maximum allowed count
                                       count = 200,
                                       include_rts = True,
                                       # Necessary to keep full_text 
                                       # otherwise only the first 140 words are extracted
                                       tweet_mode = 'extended',
                                       wait_on_rate_limit = True,
                                       wait_on_rate_limit_notify = True
                                       )
            json_data = [r._json for r in tweets]
            dfj = pd.json_normalize(json_data)
            collected_user_tweets = collected_user_tweets.append(dfj, ignore_index = True)
            sys.stdout.write("\rReading " + str(i))
            useable_ids.append(u)
        except:
            skip_ids.append(u)
            sys.stdout.write("\rReading " + str(i))
            
    pickle.dump(collected_user_tweets, open( path_to_indiana + "collected_varol-2017_human_tweets.p" , "wb"))
    print("Done")
    print(len(skip_ids), "skipped, out of", len(user_ids), "total users")
    return collected_user_tweets, useable_ids

# Loading user data example
def load_user_data():
    """
    Example of loading a json user file...
    """
    data = json.load(open(path_to_indiana + "midterm-2018_user_data.json", "r"))
    dataframe = pd.json_normalize(data)
    
def load_users():
    pass



