"""
Created on Mon Mar 15 18:28:01 2021
"""

import pickle
import pandas as pd

ferrara = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Other Datasets\\Emilio ferrara\\"

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

original = ["pandas_collected_batch_" + str(i) + ".p" for i in range(6)]

corrected= ["pandas_collected_batch_" + str(i) + "_corrected_retweets.p" for i in range(5)]

df = pd.DataFrame()
for i in range(5):
    DFO = pickle.load(open(ferrara + original[i], "rb"))
    DFC = pickle.load(open(ferrara + corrected[i], "rb"))
    DFO = DFO[~DFO["id"].isin(DFC["id"])] # i.e. dropping the re-collected data ids
    df = pd.concat([df, pd.concat([DFO, DFC], sort=False)])[extended_features]

# Adding the dataframe that I managed to collect correctly:
DFO = pickle.load(open(ferrara + original[5], "rb"))
df = pd.concat([df, DFO])[extended_features]


l = 0
for i in range(6):
    l += len(pickle.load(open(ferrara + original[i], "rb")))
print(l)
    

    
    
    