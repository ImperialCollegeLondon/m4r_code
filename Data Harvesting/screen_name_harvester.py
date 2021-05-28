"""
Attempting to retrieve screen names from user ids
"""

# US DATASET

import pandas as pd
import sys, pickle
import tweepy
import numpy as np


# *- FILE PATHS -*
collected = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Collected Tweets\\"
save_to   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Collected Tweets\\pandas_screen_names.p"


# *- SET OF UNKNOWN USER NAMES TO RETRIEVE -*
def load_unknown_user_name_ids():
    df = pickle.load(open(collected + "pandas_us_election_data.p", "rb"))
    known_names = set(df["user_id"])
    unknown_names = set(df["in_reply_to_user_id"].dropna()) | set(df["retweet_author_id"].dropna())
    unknown_names = unknown_names - known_names
    pickle.dump(unknown_names, open(collected + "user_ids_of_unknown_screen_names.p", "wb"))

# *- TWITTER API KEYS -*
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
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
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify= False)

def lookup_users(ids, api):
    full_users = pd.DataFrame()
    users_count = int(np.ceil(len(ids) / 100))
    try:
        
        for i in range(users_count):
            sys.stdout.write("\rReading batch " + str(i) + " out of " + str(users_count))
            users = api.lookup_users(user_ids = ids[i*100 : (i+1)*100])
            json_data = [u._json for u in users]
            full_users = full_users.append( pd.json_normalize(json_data) , ignore_index = True)
            #full_users.drop(labels = droplab, axis = 1, inplace = True)
        pickle.dump(full_users, open(save_to, "wb"))
        print("\n",len(full_users), "found out of", len(ids))
        return full_users
    except tweepy.TweepError:
        print('Something went wrong, quitting...')

