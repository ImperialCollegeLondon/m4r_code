"""
Title:
User Harvester

What it does:
Collects tweets based on user IDs
(i.e. to create reply or retweet networks)
(also to manually collect tweets from bot datasets)

Parameters to change before running:
n:
    maximum number of tweets to collect per user
"""
# *- NUMBER OF TWEETS TO COLLECT PER USER -*
n = 5

# *- USER ID LIST -*
# list of accounts we want to extract tweets from
user_ids = [30354991, 939091, 1249982359] # example

# *- IMPORTING PACKAGES -*
import sys, tweepy, pickle
import pandas as pd
import numpy as np
# Importing tokeniser function:
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser

# *- FILE PATHS -*
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
file  = "user_collected_tweets.p" # change


# *- TWITTER API KEYS -*
# Importing API keys (these won't appear on GitHub - see Twitter API docs)
try:
    sys.path.insert(1, m4r_data)
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

# *- METADATA/FEATURE WE WANT TO COLLECT-*
# These have the same names as those used by the Twitter API and Tweepy
# User features
user_features =  [
    'user.id',
    'user.name', 'user.screen_name', 'user.location', 'user.description',
    'user.protected', 'user.followers_count', 'user.friends_count',
    'user.listed_count', 'user.created_at', 'user.favourites_count',
    'user.utc_offset', 'user.geo_enabled',
    'user.verified', 'user.statuses_count', 'user.lang',
    'user.default_profile', 'user.default_profile_image'
    ]
# Tweet metadata
features = [
    'created_at', 'id', 'full_text',
    'in_reply_to_status_id', 'in_reply_to_user_id', 'in_reply_to_screen_name', 'is_quote_status',
    'retweet_count', 'favorite_count', 'lang',
    'entities.hashtags', 'entities.symbols',
    'entities.user_mentions', 'entities.urls'] + user_features
# Additional Tweet Metadata: specifically if the tweet is a retweet
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

if n > 0:
    # *- INITIALISING -*
    df_entire = pd.DataFrame()
    
    # *- BEGINNING RETRIEVAL -*
    for i, ID in enumerate(user_ids):
        # retrieving the i-th user_id
        statuses = api.user_timeline(user_id = ID, count = n, include_rts = True, tweet_mode = 'extended', wait_on_rate_limit = True, wait_on_rate_limit_notify = True)
        json_data = [r._json for r in statuses] # Converting to JSON data
        dfj = pd.json_normalize(json_data) # Converting to a pandas dataframe
        try:
            dfj = dfj[extended_features] # Keeping only relevant features (retweet)
        except:
            dfj = dfj[features] # Keeping only relevant features (non-retweet)
        df_entire = df_entire.append(dfj, ignore_index = True) # appending batch of tweets to dataframe
        sys.stdout.write("\rReading " + str(i+1) + " out of " + str(len(user_ids)))
    
    # *- PROCESSING FEATURES -*
    # Retrieving index of tweets that are not retweets:
    nonrt_index = df_entire[df_entire["retweeted_status.id"].isna()].index
    # Retrieving index of tweets that are retweets:
    rt_index    = df_entire[df_entire["retweeted_status.id"].isna() == False].index
    # Retrieving counts for hashtags, mentions, and urls for non retweets:
    df_entire.loc[nonrt_index, "hashtag_count"] = df_entire.loc[nonrt_index, "entities.hashtags"].str.len() # counting number of hashtags in tweet
    df_entire.loc[nonrt_index, "mention_count"] = df_entire.loc[nonrt_index, "entities.user_mentions"].str.len() # counting number of mentions in tweet
    df_entire.loc[nonrt_index, "url_count"]     = df_entire.loc[nonrt_index, "entities.urls"].str.len() # counting number of urls in tweet
    df_entire.loc[nonrt_index, "tokenised_text"]= df_entire.loc[nonrt_index, "full_text"].apply(lambda x: text_tokeniser(x)) # tokenising the tweet
    # Retrieving counts for hashtags, mentions, and urls for retweets:
    # We have to separately do this because retweet full texts are truncated
    df_entire.loc[rt_index, "hashtag_count"] = df_entire.loc[rt_index, "retweeted_status.entities.hashtags"].str.len() # counting number of hashtags in tweet
    df_entire.loc[rt_index, "mention_count"] = df_entire.loc[rt_index, "retweeted_status.entities.user_mentions"].str.len() + 1 # counting number of mentions in tweet and adjusting by adding 1 (since RT @user:)
    df_entire.loc[rt_index, "url_count"]     = df_entire.loc[rt_index, "retweeted_status.entities.urls"].str.len() # counting number of urls in tweet
    df_entire.loc[rt_index, "tokenised_text"]= "rt <user> : " + df_entire.loc[rt_index, "retweeted_status.full_text"].apply(lambda x: text_tokeniser(x)) # tokenising the tweet
    # "Un-truncating" the retweeted status
    df_entire.loc[rt_index, "full_text"]     = df_entire.loc[rt_index, "full_text"].str.split().apply(lambda x: str(x[0]) + " " + str(x[1]) + " ") + df_entire.loc[rt_index, "retweeted_status.full_text"]
    
    # pickle.dump(df_entire, open(folder + file, "wb")) # Saving dataframe
    
elif n == 0:
    # *- INITIALISING -*
    df_entire = pd.DataFrame()
    num_batches = int(np.ceil(len(user_ids) / 100))
    
    # *- BEGINNING RETRIEVAL -*
    for i in range(num_batches):
        try:
            sys.stdout.write("\rRetrieving batch " + str(i) + " out of " + str(num_batches))
            users = api.lookup_users(user_ids = user_ids[i*100 : (i+1)*100])
            json_data = [u._json for u in users]
            dfj = pd.json_normalize(json_data)
            df_entire = df_entire.append( dfj[user_features] , ignore_index = True)
            #full_users.drop(labels = droplab, axis = 1, inplace = True)
        except:
            pass
    print("\n",len(df_entire), "found out of", len(set(user_ids)))

    pickle.dump(df_entire, open(m4r_data + file, "wb"))
    
    

