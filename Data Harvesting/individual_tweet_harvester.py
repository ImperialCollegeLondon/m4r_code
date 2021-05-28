"""
Title:
Tweet Harvester

What it does:
Collects tweets based on Tweet IDs

EMILIO FERRARA DATASET

(i.e. using the )

Parameters to change before running:
overwrite:
    True => overwrites the file us_election_data.csv
    False => appends the data to the file us_election_data.csv
new_header:
    True => creates a new header row (automatically True if overwriting)
    False => doesn't create a new header row (leave False if overwrite = False)
n:
    maximum number of tweets to collect upon running the code
    set to None if you want to max out at the Twitter rate limit
"""


features = [
    'created_at', 'id', 'full_text',
    'in_reply_to_status_id',
    'in_reply_to_user_id',
    'in_reply_to_screen_name',
    'is_quote_status', # 'quoted_status_id',
    'retweet_count',
    'favorite_count',#'possibly_sensitive',
    'lang', 'entities.hashtags', 'entities.symbols',
    'entities.user_mentions', 'entities.urls', 'user.id',
    'user.name', 'user.screen_name', 'user.location', 'user.description',
    'user.protected', 'user.followers_count', 'user.friends_count',
    'user.listed_count', 'user.created_at', 'user.favourites_count',
    'user.utc_offset', 'user.geo_enabled',
    'user.verified', 'user.statuses_count', 'user.lang',
    'user.default_profile', 'user.default_profile_image'
    ]

extended_features = features + [
    "retweeted_status.id",
    "retweeted_status.full_text",
    'retweeted_status.entities.hashtags',
    'retweeted_status.entities.symbols',
    'retweeted_status.entities.user_mentions',
    'retweeted_status.entities.urls',
    'retweeted_status.user.id',
    'retweeted_status.user.screen_name'
    ]

# *- PACKAGES -*
import tweepy
import sys, os
from datetime import date
import csv
import time
import pandas as pd
import pickle

# *- FILE PATHS -*
ferrara = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Other Datasets\\Emilio ferrara\\"
folder_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting"
#file_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Collected Tweets\\us_election_data.csv" # for us_elections_data.csv
file_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Collected Tweets\\georgia_election_data.csv"
cwd = os.getcwd()
if folder_path not in sys.path:
    sys.path.append(folder_path)

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


# f = "us-presidential-tweet-id-2020-11-01-" # THESE WERE DONE POORLY - THE RETWEET STATUSES MAY BE BAD
# f = "us-presidential-tweet-id-2020-11-02-" # THESE WERE DONE POORLY - THE RETWEET STATUSES MAY BE BAD
# f = "us-presidential-tweet-id-2020-11-03-" # THESE WERE DONE POORLY - THE RETWEET STATUSES MAY BE BAD
# f = "us-presidential-tweet-id-2020-11-04-" # THESE WERE DONE POORLY - THE RETWEET STATUSES MAY BE BAD
f = "us-presidential-tweet-id-2020-11-05-"
G = ["0" + str(x) for x in range(10)] + [str(x) for x in range(10,24)]



# # Sampling the IDs that I messed up...
# df = pickle.load(open(ferrara + "pandas_collected_batch_4.p", "rb"))
# IDS = df[(df.full_text.str.contains("RT @")) & df.full_text.str.contains("…")].id.tolist()

# Sampling the IDs...
IDS = pd.DataFrame(columns = ["id"])
ng = 10000
for i,g in enumerate(G):
    g_ = pd.read_csv(ferrara + f + g + ".txt", header = None)
    g_.columns = ["id"]
    IDS = pd.concat([IDS, g_.sample(n = ng, random_state = 1349565 // (i + 3) )], ignore_index = True)
IDS = IDS.id.tolist()

IDS = [IDS[i:i+100] for i in range(0, len(IDS), 100)] # batch into sublists of 100...


# Retrieving the tweets from the samples...
df_entire = pd.DataFrame()
for i, ID in enumerate(IDS):
    statuses = api.statuses_lookup(ID, tweet_mode = "extended", wait_on_rate_limit = True, wait_on_rate_limit_notify = True) 
    json_data = [r._json for r in statuses]
    dfj = pd.json_normalize(json_data)
    try:
        dfj = dfj[extended_features]
    except:
        dfj = dfj[features]
    df_entire = df_entire.append(dfj, ignore_index = True)
    sys.stdout.write("\rReading " + str(i))


# pickle.dump(df_entire, open(ferrara + "pandas_collected_batch_2.p", "wb"))