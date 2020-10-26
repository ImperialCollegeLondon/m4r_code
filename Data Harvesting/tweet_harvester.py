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
# *- OVERWRITE? -*
overwrite = False

# *- INITIALISE HEADER? -*
new_header = False

# *- NUMBER OF TWEETS TO COLLECT? -*
n = 5000









# *- PACKAGES -*
import tweepy
import sys, os
from datetime import date
import csv
import time











# *- FILE PATHS -*
folder_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting"
file_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data\\us_election_data.csv"
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











# Importing tokeniser function:
from tweet_tokeniser import tweet_tokeniser









# *- SEARCH PARAMETERS -*
today = str(date.today())
count = 100
search_terms_1 = "#trump OR #donaldtrump OR #trump2020 OR #votetrump OR #trumppence2020 OR #gop OR #republicans OR "
search_terms_2 = "#biden OR #joebiden OR #biden2020 OR #votebiden OR #bidenharris2020 OR #votedems OR #democrats OR "
search_terms_3 = "#uselection OR #uselection2020 OR #presidentialelection2020 OR #elections2020 OR #potus OR #whitehouse"
search_terms = search_terms_1 + search_terms_2 + search_terms_3











# *- NUMBER OF TWEETS TO COLLECT -*
#n = None
nt = 0










# *- TWEEPY ITERATOR OBJECT -*
c = tweepy.Cursor(api.search, q=search_terms, lang='en', since=today, count=count, tweet_mode='extended').items()













# Boolean object to determine when to stop
stop = False






# Data Labels
tweet_headers = [
    'full_text',
    'tokenised_text',
    'tweet_id',
    'user_id',
    'created_at'
    ]

tweet_metadata_headers = [
    'retweet_count',
    'favourite_count',
    'length_of_tweet',
    'num_hashtags',
    'num_urls',
    'num_media',
    'num_mentions',
    'num_emojis',
    'is_retweet',
    'is_quote_status',
    'tweet_source'
    ]

user_headers = [
    'user_screen_name',
    'user_name',
    'user_is_verified',
    'user_created_at',
    'user_followers_count',
    'user_friends_count',
    'user_listed_count',
    'user_favourites_count',
    'user_statuses_count',
    'user_default_profile',
    'user_default_profile_image',
    'user_default_profile_banner',
    'user_description',
    'user_location',
    'user_url',
    'user_profile_banner_url',
    'user_profile_image_url',
    'user_lang',
    'user_time_zone',
    'user_geo_enabled',
    'user_contributors_enabled',
    'user_has_extended_profile',
    ]

tweet_interaction_headers = [
    'in_reply_to_status_id',
    'in_reply_to_user_id',
    'retweet_status_id',
    'retweet_author_id',
    'quote_status_id',
    ]

header = tweet_headers + tweet_metadata_headers + user_headers + tweet_interaction_headers










# *- CHECKING THAT THE FILE EXISTS
try:
    with open(file_path, 'x', newline='', encoding = 'utf-8') as w:
        writer = csv.writer(w)
        writer.writerow(header)
except:
    pass








# *- COLLECTING AND RECORDING TWEETS -*
mode = 'w' if overwrite else 'a'
with open(file_path, mode=mode, newline='', encoding="utf-8") as w:
    writer = csv.writer(w)
    if new_header or overwrite:
        writer.writerow(header)
    while not stop:
        try:
            # Retrieving the next tweet in the iterator
            tweet = next(c)
            
            # Retweet?
            try:
                actual_status = tweet.retweeted_status
                is_rt = True
            except:
                actual_status = tweet
                is_rt = False
                
            full_text = actual_status.full_text
            
            num_hashtags = len(actual_status.entities["hashtags"])
            
            # Collecting URL data
            url_data = []
            for u in actual_status.entities["urls"]:
                url_data.append([u["url"], u["indices"][0], u["indices"][1]])
            if "media" in actual_status.entities:
                num_media = len(actual_status.entities["media"])
                for u in actual_status.entities["media"]:
                    url_data.append([u["url"], u["indices"][0], u["indices"][1]])
            else:
                num_media = 0
            
            
            # Collecting user_mentions_data
            user_mentions_data = []
            for u in actual_status.entities["user_mentions"]:
                user_mentions_data.append(u["screen_name"])
                
            # Tokenising the tweet
            tokenised_text, num_hashtags, num_mentions, num_urls, num_emojis = tweet_tokeniser(full_text, url_data, user_mentions_data)
            
            # Gathering interaction data
            rt_id = None
            rt_user = None
            qt_id = None
            if is_rt:
                rt_id = actual_status.id
                rt_user = tweet.author.id
            if tweet.is_quote_status:
                try:
                    qt_id = tweet.quoted_status_id
                except:
                    qt_id = None
                    
                
            
            # Checking if there is a profile banner or not:
            try:
                user_prof_banner = tweet.user.profile_banner_url
                user_default_banner = True
            except:
                user_prof_banner = None
                user_default_banner = False
            
            
            # *- COMPILING THE DATA -*
            tweet_data = [
                full_text,
                tokenised_text,
                tweet.id,
                tweet.user.id,
                tweet.created_at
                ]
            
            tweet_metadata = [
                tweet.retweet_count,
                tweet.favorite_count,
                tweet.display_text_range[1],
                num_hashtags,
                num_urls - num_media,
                num_media,
                num_mentions,
                num_emojis,
                is_rt,
                tweet.is_quote_status,
                tweet.source
                ]
            
            tweet_interaction_data = [
                tweet.in_reply_to_status_id,
                tweet.in_reply_to_user_id,
                rt_id,
                rt_user,
                qt_id
                ]
            
            user_data = [
                tweet.user.screen_name,
                tweet.user.name,
                tweet.user.verified,
                tweet.user.created_at,
                tweet.user.followers_count,
                tweet.user.friends_count,
                tweet.user.listed_count,
                tweet.user.favourites_count,
                tweet.user.statuses_count,
                tweet.user.default_profile,
                tweet.user.default_profile_image,
                user_default_banner,
                tweet.user.description,
                tweet.user.location,
                tweet.user.url,
                user_prof_banner,
                tweet.user.profile_image_url_https,
                tweet.user.lang,
                tweet.user.time_zone,
                tweet.user.geo_enabled,
                tweet.user.contributors_enabled,
                tweet.user.has_extended_profile
                ]
            


            row = tweet_data + tweet_metadata + user_data + tweet_interaction_data
            writer.writerow(row)
            nt += 1
            if nt == n:
                print(n, "tweets collected, stopping")
                break
        # handle rate limit by waiting till it resets
        except tweepy.TweepError:
            print("max rate hit, sleeping, press any key to interrupt,\n", nt, "tweets collected so far" )
            for i in range(60):
                try:
                    time.sleep(15)
                except KeyboardInterrupt:
                    stop = True
                    break
            continue
        except StopIteration:
            print(nt, "tweets collected, end of search")
            break
