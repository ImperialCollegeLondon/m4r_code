"""
Created on Wed Feb 17 07:39:53 2021

@author: fangr
"""

# *- MODULES -*
import sys, pickle
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Importing functions to create Contextual LSTM layers:
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.backend import cast


# Local modules
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection")
from mixed_contextual_lstm_02_preprocessing import *
from mixed_contextual_lstm_03_training import *


# *- FILE PATHS -*
indiana = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Indiana Dataset\\"
italy   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Cresci Italian Dataset\\"
lstmdata= "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\LSTM Training Data\\"
collected = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Collected Tweets\\"
dset = ["astroturf.p", "cresci-rtbust-2019.p", "gilani-2017.p", "midterm_2018.p", "political-bots-2019.p", "social_spambots_1.p", "social_spambots_2.p", "social_spambots_3.p", "traditional_spambots_1.p", "varol-2017.p", "genuine_accounts.p", "gilani-2017.p"]
dset = ["bot_sample_" + dset[x] if x < 10 else "human_sample_" + dset[x] for x in range(12) ]
glove_file = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
weightsavepath = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection\\mixed_contextual_LSTM_weights_1\\"

def load_collected_data():
    """
    lOADING US PRES ELECTION DATA
    """
    df = pickle.load(open(collected + "pandas_us_election_data.p", "rb"))
    ri = df.index[df["is_retweet"]]
    df.iloc[ri,1] = df.iloc[ri,1].apply(lambda x : ['<allcaps>', 'rt', '<user>', ':'] + x) # adding in the "rt @USER:" before retweets
    
    tweet_columns = ["tweet", "num_hashtags", "num_mentions", "num_urls", 'retweet_count', 'favorite_count', "len_tweet"]
    user_columns_count = ["user.age", "user.followers_count", "user.friends_count", "user.statuses_count", "user.favourites_count", "user.listed_count"]
    user_columns_bool = ["user.default_profile", "user.default_profile_image", "user.geo_enabled", "user.verified"] # , "user.has_location"
    derived_columns = ["user.len_name", "user.len_screen_name", "user.num_digits_name", "user.num_digits_screen_name", "user.len_description"]
    class_columns = ["full_text", "id", "user.id", "created_at"]
    vader_columns = ['vader_compound', 'vader_neg', 'vader_neu', 'vader_pos']
    relevant_columns = tweet_columns + user_columns_count + user_columns_bool + derived_columns + class_columns + vader_columns
    # ===================
    # AGE: Calculating the age of the account
    # ===================
    harvest_date = datetime.datetime(2020,11,3,0,0,0) # FOR THE US PRES: 3RD NOV 2020 
    df["user.age"] = (harvest_date - pd.to_datetime(df["user_created_at"], format='%Y-%m-%d %H:%M:%S')).dt.days
    # ===================
    # DIGITS IN SCREENNAME ETC...
    # ===================
    df["user.len_screen_name"] = df["user_screen_name"].str.len()
    df["user.len_name"] = df["user_name"].str.len()
    df["user.num_digits_screen_name"] = df["user_screen_name"].str.count(r"\d")
    df["user.num_digits_name"] = df["user_name"].str.count(r"\d")
    # ===================
    # LENGTH OF TWEET AND DESCRIPTION
    # ===================
    df["len_tweet"] = df["length_of_tweet"]
    df["user.len_description"] = df["user_description"].str.len()
    # ===================
    # CHANGE COLUMN NAMES
    # ===================
    rd1 = {'tweet_id': 'id',"user_id" : "user.id",'tokenised_text': 'tweet',"favourite_count":"favorite_count"}
    rd2 = {"user_is_verified" : "user.verified",**{"user_" + x[5:] : x for x in user_columns_count}, **{"user_" + x[5:] : x for x in user_columns_bool}}
    df.rename(columns= {**rd1, **rd2}, inplace=True)
    df = df[relevant_columns]
    df = df.fillna(0)
    return df

def vectorizing_collected(vectorizer):
    # Vectorizing collected data in batches
    vectorized_collected_data_batches = []
    r = int(np.ceil(len(df)/128))
    print(" ")
    for i in range(r):
        sys.stdout.write("\r" + str(i+1) + " out of " + str(r))
        vectorized_collected_data_batches.append(vectorizer(df["tweet"].str.join(" ")[128*i:128*(i+1)].tolist()))
    us_tweet_vector = tf.concat(vectorized_collected_data_batches, axis = 0)
    print(" ")
    # Retrieving metadata
    us_metadata = df.iloc[:,1:22].to_numpy()
    
    

def run_model_on_collected_tweets(vectorizer):
    df = load_collected_data()
    us_metadata = df.iloc[:,1:22].to_numpy().astype('float32')
    model = mixed_contextual_lstm_model()
    model.load_weights(weightsavepath)
    lstm_p = model.predict({"tweet_input" : us_tweet_vector, "metadata_input": us_metadata.astype("float32")})
    # The output is a 2 x no.tweets tensor thing
    # I think the first one is the final output, and the second one is the auxiliary output
    final = np.round(lstm_p[:][0]).reshape((-1,))
    aux   = np.round(lstm_p[:][1]).reshape((-1,))
    df["lstm_predict"] = np.round(lstm_p[:][0]).reshape((-1,))
    df["lstm_predict_aux"] = np.round(lstm_p[:][1]).reshape((-1,))
    
    no_humans = len(df[df["lstm_predict"] == 0])
    no_bots   = len(df[df["lstm_predict"] == 1])
    x_w = np.empty((no_humans,))
    x_w.fill(1/no_humans)
    y_w = np.empty((no_bots,))
    y_w.fill(1/no_bots)
    #plt.hist([df[df["rfc_prediction"] == 0]["vader_compound"], df[df["rfc_prediction"] == 1]["vader_compound"]], density = True, label = ["Human", "Bot"]);
    plt.hist([df[df["lstm_predict"] == 0]["vader_compound"], df[df["lstm_predict"] == 1]["vader_compound"]], weights=[x_w, y_w], label = ["Human", "Bot"]);
    plt.title("Comparing Compound Vader Sentiment Score Across US Election Tweets\n(Contextual LSTM)")
    plt.xlabel("Compound Vader Sentiment Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    
    
    # For the following have to run account level mixed as welll....
    print("Number of tweets RFC and LSTM DO agree on:")
    print(len(df[df["lstm_predict"] == df["rfc_predict"]]))
    print("Number of tweets RFC and LSTM DON'T agree on:")
    print(len(df[df["lstm_predict"] != df["rfc_predict"]]))
    print(len(df[df["lstm_predict"] == df["rfc_predict"]]) / len(df) * 100, "% agreement")
    
    print("Number of tweets LSTM and LSTM Aux DO agree on:")
    print(len(df[df["lstm_predict"] == df["lstm_predict_aux"]]))
    print("Number of tweets LSTM and LSTM Aux DON'T agree on:")
    print(len(df[df["lstm_predict"] != df["lstm_predict_aux"]]))
    print(len(df[df["lstm_predict"] == df["lstm_predict_aux"]]) / len(df) * 100, "% agreement")
    
    # US ELECTION DATA
    earliest = df["created_at"].sort_values().iloc[0].date()
    latest = df["created_at"].sort_values().iloc[-1].date()
    r = pd.date_range(earliest, latest, freq='1D')
    
    df_hum = df[df["lstm_predict"] == 0][["created_at", "vader_compound"]]
    df_hum["date"] = df_hum["created_at"].apply(lambda x: x.date())
    plt.plot(df_hum.groupby(['date']).mean(), label = "Human");
    
    df_bot = df[df["lstm_predict"] == 1][["created_at", "vader_compound"]]
    df_bot["date"] = df_bot["created_at"].apply(lambda x: x.date())
    plt.plot(df_bot.groupby(['date']).mean(), label = "Bot");
    
    plt.xticks(rotation = 45);
    plt.title("Compound Vader Sentiment for US Election Tweets (Grouped By Days)\n With Bot/Human Predictions");
    plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.5)
    plt.legend()
    plt.show()
    
    plt.plot(df_reduced.groupby(pd.Grouper(key = 'created_at', freq = "0.25D")).mean().dropna(how = "any")); plt.xticks(rotation = 45);
    plt.title("Compound Vader Sentiment for US Election Tweets (Grouped By 1/4 Days)");
    plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.5)
    plt.show()
    
    # GEORGIA ELECTION DATA    
    earliestg = dfg["created_at"].sort_values().iloc[0].date()
    latestg = dfg["created_at"].sort_values().iloc[-1].date()
    rg = pd.date_range(earliestg, latestg, freq='1D')
    plt.hist(dfg["created_at"], rwidth = 0.8, bins = rg); plt.xticks(rotation = 45);
    plt.title("Distribution of Collected Georgia Election Tweets (Grouped By Days)"); plt.show()
    
    df_reducedg = dfg[["created_at", "vader_compound"]]
    df_reducedg["date"] = df_reducedg["created_at"].apply(lambda x: x.date())
    plt.plot(df_reducedg.groupby(['date']).mean()); plt.xticks(rotation = 45);
    plt.title("Compound Vader Sentiment for Georgia Election Tweets (Grouped By Days)");
    plt.axvspan(datetime.datetime(2021, 1, 5, 0, 0), datetime.datetime(2021,1,6,0,0),color = "red", alpha = 0.5)
    plt.show()
    
    plt.plot(df_reducedg.groupby(pd.Grouper(key = 'created_at', freq = "0.25D")).mean().dropna(how = "any")); plt.xticks(rotation = 45);
    plt.title("Compound Vader Sentiment for Georgia Election Tweets (Grouped By 1/4 Days)");
    plt.axvspan(datetime.datetime(2021, 1, 5, 0, 0), datetime.datetime(2021,1,6,0,0),color = "red", alpha = 0.5)
    plt.show()
    
    df_hum_2 = df_hum[(df_hum["created_at"] > datetime.datetime(2020, 11, 1, 0, 0)) & (df_hum["created_at"] < datetime.datetime(2020,11,6,0,0))]
    plt.plot(df_hum_2.groupby(pd.Grouper(key = 'created_at', freq = "H")).mean().dropna(how = "any") , marker = "x", label = "Human");
    df_bot_2 = df_bot[(df_bot["created_at"] > datetime.datetime(2020, 11, 1, 0, 0)) & (df_bot["created_at"] < datetime.datetime(2020,11,6,0,0))]
    plt.plot(df_bot_2.groupby(pd.Grouper(key = 'created_at', freq = "H")).mean().dropna(how = "any") , marker = "x", label = "Bot");
    plt.legend()
    plt.xticks(rotation = 45);
    plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.5)
    plt.title("Compound Vader Sentiment for US Election Tweets\n (Grouped By Hours, 2 Days before and after election day)")
    plt.show()
    
    
    