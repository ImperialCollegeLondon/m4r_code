"""
Title:
Tweet Harvester

What it does:
Collects tweets based on election hashtag searchterms

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

# *- NUMBER OF TWEETS TO COLLECT -*
n = 5

# *- IMPORTING PACKAGES -*
import sys, time, tweepy, pickle
from datetime import date
import pandas as pd
# Importing tokeniser function:
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser

# *- FILE PATHS -*
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
us_file  = m4r_data + "us_election_data.p" # file save path
#ga_file  = m4r_data + "ga_election_data.p"


# *- TWITTER API KEYS -*
# Importing API keys (these won't appear on GitHub - see Twitter API docs)
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

# *- SEARCH PARAMETERS -*
# Search terms for the US Election:
search_terms = "#trump OR #donaldtrump OR #trump2020 OR #votetrump OR #trumppence2020 OR #gop OR #republicans OR #biden OR #joebiden OR #biden2020 OR #votebiden OR #bidenharris2020 OR #votedems OR #democrats OR #uselection OR #uselection2020 OR #presidentialelection2020 OR #election2020 OR #potus OR #whitehouse"
# # Alternative search terms for the Georgia Election:
# search_terms = "#georgia OR #gapol OR #ossoff OR #warnock OR #perdue OR #loeffler OR #georgiarunoffs OR #gasen OR #gasenateraces OR @ossoff OR @sendavidperdue OR @kloeffler OR @reverendwarnock"
today = str(date.today()) # today's date and time (date and time for the search)
count = 100 # number of tweets to search for each iteration of the Tweepy Cursor

# *- METADATA/FEATURE WE WANT TO COLLECT-*
# These have the same names as those used by the Twitter API and Tweepy
features = [
    'created_at', 'id', 'full_text',
    'in_reply_to_status_id', 'in_reply_to_user_id', 'in_reply_to_screen_name', 'is_quote_status',
    'retweet_count', 'favorite_count', 'lang',
    'entities.hashtags', 'entities.symbols',
    'entities.user_mentions', 'entities.urls', 'user.id',
    'user.name', 'user.screen_name', 'user.location', 'user.description',
    'user.protected', 'user.followers_count', 'user.friends_count',
    'user.listed_count', 'user.created_at', 'user.favourites_count',
    'user.utc_offset', 'user.geo_enabled',
    'user.verified', 'user.statuses_count', 'user.lang',
    'user.default_profile', 'user.default_profile_image'
    ]
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

# *- TWEEPY ITERATOR OBJECT -*
# Searches for tweets corresponding to search terms - up to 100 at a time
c = tweepy.Cursor(api.search, q=search_terms, lang='en', since=today, count=count, tweet_mode='extended').items()

# *- INITIALISING -*
stop = False # Boolean object to determine when to stop the search
nt = 0 # Count object to keep track of how many tweets we have collected so far
df_entire = pd.DataFrame() # Pandas Dataframe to store tweets and metadata

# *- BEGINNING THE SEARCH -*
while not stop:
    try:
        tweet = next(c) # Retrieving the next tweet in the iterator
        dfj = pd.json_normalize(tweet._json) # Converting to a dataframe
        try: # Keeping only relevant features (if it is a retweet)
            dfj = dfj[extended_features]
        except: # Keeping only relevant features (if it is NOT a retweet)
            dfj = dfj[features]
        df_entire = df_entire.append(dfj, ignore_index = True) # Appending to the main dataframe
        nt += 1 # Updating count
        if nt == n: # If we have collected enough tweets, we stop
            print(n, "tweets collected, stopping")
            break
    # Handling rate limit - wait up to 15 mins before attempting to search again
    except tweepy.TweepError:
        print("max rate hit, sleeping, press any key to interrupt,\n", nt, "tweets collected so far" )
        for i in range(60):
            try:
                time.sleep(15)
            except KeyboardInterrupt:
                stop = True
                break
        continue
    except StopIteration: # Rate Limit Exceeded and won't refresh - simply stop the search
        print(nt, "tweets collected, end of search")
        break

# *- PROCESSING FEATURES -*

df_entire["hashtag_count"] = df_entire["entities.hashtags"].str.len() # counting number of hashtags in tweet
df_entire["mention_count"] = df_entire["entities.user_mentions"].str.len() # counting number of mentions in tweet
df_entire["url_count"]     = df_entire["entities.urls"].str.len() # counting number of urls in tweet
df_entire["tokenised_text"]= df_entire["full_text"].apply(lambda x: text_tokeniser(x)) # tokenising the tweet

try:
    df = pickle.load(open(us_file, "rb")) # Loading dataframe of previously collected tweets
    # df = pickle.load(open(ga_file, "rb"))
except:
    df = pd.DataFrame()
    
df = pd.concat([df, df_entire], ignore_index = True) # Appending to previously collected tweets

# pickle.dump(df, open(us_file, "wb")) # Saving dataframe
# pickle.dump(df, open(ga_file, "wb"))




