"""
Title: Sentiment Analysis

Description: Performing Sentiment Analysis using VADER
"""

# 1. SETUP --------------------------------------------------------------------
import pickle, sys, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sns.set(font="Arial") # Setting the plot style
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\" # folder location where data is held
# Importing Account Level Detection
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_repository")
from account_level_detection import get_full_dataset, features
figure_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Report\\Figures\\" # folder location to store files


def distribution_of_vader_sentiment():
    """
    Comparing the distributions of the Vader scores over the Training,
    US, and Georgia datasets.
    """
    df = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    us = df[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    df = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    ga = df[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    df = pickle.load(open(m4r_data + "balanced_tweet_training_data.p", "rb"))
    trn = df[["user.id", "full_text", "tokenised_text", 'vader', 'class']]
    df = None
    
    #=-=-=-=-=--=-=--=-=-=-=-=-===-=-=-=-=-=-=-=-=-=-==-=-=-=-=--=-=--=-=-=-=-=
    # PLOT 1: UNADJUSTED
        
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), sharey = True)
    fig.suptitle('Distribution of VADER Scores', fontweight = "bold")
    # TRAINING
    sns.histplot(ax = axes[0],
                 data = trn.sort_values("class", ascending = False),
                 x = "vader",
                 hue = "class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step",
        )
    axes[0].set_title("Training", fontweight = "bold");
    axes[0].set_xlabel("")
    
    # US
    sns.histplot(ax = axes[1],
                 data = us.sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[1].set_title("US", fontweight = "bold");
    axes[1].set_xlabel("VADER Score", fontweight = "bold")
    axes[1].get_legend().remove()
    # GEORGIA
    sns.histplot(ax = axes[2],
                 data = ga.sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[2].set_title("Georgia", fontweight = "bold");
    axes[2].set_xlabel("")
    axes[2].get_legend().remove()
    
    axes[0].set_ylabel("Density", fontweight = "bold")
    
    old_legend = axes[0].legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    axes[1].legend(handles, labels, ncol = 2, loc='upper center', bbox_to_anchor=(-0.62, -0.13))
    axes[0].get_legend().remove()
    plt.subplots_adjust(wspace = 0.02, hspace = 0.3, top = 0.8)
    
    plt.savefig(figure_path + "compare_vader_polarity_scores.pdf", bbox_inches = "tight")
    
    plt.show()
    
    #=-=-=-=-=--=-=--=-=-=-=-=-===-=-=-=-=-=-=-=-=-=-==-=-=-=-=--=-=--=-=-=-=-=
    # PLOT 2: NEUTRAL SCORES REMOVED
        
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), sharey = True)
    fig.suptitle('Distribution of VADER Scores: Neutral Scores Removed', fontweight = "bold")
    # TRAINING
    sns.histplot(ax = axes[0],
                 data = trn[trn["vader"] != 0].sort_values("class", ascending = False),
                 x = "vader",
                 hue = "class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step",
        )
    axes[0].set_title("Training", fontweight = "bold");
    axes[0].set_xlabel("")
    
    # US
    sns.histplot(ax = axes[1],
                 data = us[us["vader"] != 0].sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[1].set_title("US", fontweight = "bold");
    axes[1].set_xlabel("VADER Score", fontweight = "bold")
    axes[1].get_legend().remove()
    # GEORGIA
    sns.histplot(ax = axes[2],
                 data = ga[ga["vader"] != 0].sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[2].set_title("Georgia", fontweight = "bold");
    axes[2].set_xlabel("")
    axes[2].get_legend().remove()
    
    axes[0].set_ylabel("Density", fontweight = "bold")
    
    old_legend = axes[0].legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    axes[1].legend(handles, labels, ncol = 2, loc='upper center', bbox_to_anchor=(-0.62, -0.13))
    axes[0].get_legend().remove()
    plt.subplots_adjust(wspace = 0.02, hspace = 0.3, top = 0.8)
    
    plt.savefig(figure_path + "compare_vader_polarity_scores_neutral_removed.pdf", bbox_inches = "tight")
    
    plt.show()
    
    
    
    #=-=-=-=-=--=-=--=-=-=-=-=-===-=-=-=-=-=-=-=-=-=-==-=-=-=-=--=-=--=-=-=-=-=
    # PLOT 3: CONCATENATING TWEETS
    
    # CONCATENATING TWEETS
    N_TWEETS = 5
    
    # finding number of users with at least 5 tweets...
    print("TRN users with at least " + str(N_TWEETS) + " tweets: ", sum(trn.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    print("US users with at least " + str(N_TWEETS) + " tweets:  ", sum(us.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    print("GA users with at least " + str(N_TWEETS) + " tweets:  ", sum(ga.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    # picking out user ids with at least N_TWEETS tweets
    trn_available_users = (trn.groupby(["user.id"]).first()[trn.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    us_available_users = (us.groupby(["user.id"]).first()[us.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    ga_available_users = (ga.groupby(["user.id"]).first()[ga.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    # joining the firt N_TWEETS tweets together to form a corpus for the dataset
    trn2 = (trn[trn["user.id"].isin(trn_available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    us2 = (us[us["user.id"].isin(us_available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    ga2 = (ga[ga["user.id"].isin(ga_available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    
    
    # Applying Vader analysis to it (doesn't take too long)
    sid = SentimentIntensityAnalyzer()
    
    trn2["vader"] = trn2["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    trn2 = trn2.merge(trn.groupby(["user.id"]).first().reset_index()[["user.id", "class"]], how = "left", on = "user.id")
    
    us2["vader"] = us2["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    us2 = us2.merge(us.groupby(["user.id"]).first().reset_index()[["user.id", "predicted_class"]], how = "left", on = "user.id")
    
    ga2["vader"] = ga2["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    ga2 = ga2.merge(ga.groupby(["user.id"]).first().reset_index()[["user.id", "predicted_class"]], how = "left", on = "user.id")
    
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), sharey = True)
    fig.suptitle('Distribution of VADER Scores: 5 Tweet Corpus', fontweight = "bold")
    # TRAINING
    sns.histplot(ax = axes[0],
                 data = trn2.sort_values("class", ascending = False),
                 x = "vader",
                 hue = "class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step",
        )
    axes[0].set_title("Training", fontweight = "bold");
    axes[0].set_xlabel("")
    
    # US
    sns.histplot(ax = axes[1],
                 data = us2.sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[1].set_title("US", fontweight = "bold");
    axes[1].set_xlabel("VADER Score", fontweight = "bold")
    axes[1].get_legend().remove()
    # GEORGIA
    sns.histplot(ax = axes[2],
                 data = ga2.sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[2].set_title("Georgia", fontweight = "bold");
    axes[2].set_xlabel("")
    axes[2].get_legend().remove()
    
    axes[0].set_ylabel("Density", fontweight = "bold")
    
    old_legend = axes[0].legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    axes[1].legend(handles, labels, ncol = 2, loc='upper center', bbox_to_anchor=(-0.62, -0.13))
    axes[0].get_legend().remove()
    plt.subplots_adjust(wspace = 0.02, hspace = 0.3, top = 0.8)
    
    plt.savefig(figure_path + "compare_vader_polarity_scores_5_tweets_concat.pdf", bbox_inches = "tight")
    
    plt.show()
    
    
    #=-=-=-=-=--=-=--=-=-=-=-=-===-=-=-=-=-=-=-=-=-=-==-=-=-=-=--=-=--=-=-=-=-=
    # PLOT 4: REPLIES ONLY
        
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9), sharey = True, sharex = True)
    fig.suptitle('Distribution of VADER Scores: Replies + Neutral Removed', fontweight = "bold")
    # US
    sns.histplot(ax = axes[0],
                 data = us[(us["in_reply_to_status_id"].isna() == False) & (us["vader"] != 0)].sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[0].set_title("US", fontweight = "bold");
    axes[0].set_xlabel("")
    # GEORGIA
    sns.histplot(ax = axes[1],
                 data = ga[(ga["in_reply_to_status_id"].isna() == False) & (ga["vader"] != 0)].sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "predicted_class",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step"
        )
    axes[1].set_title("Georgia", fontweight = "bold");
    axes[1].set_xlabel("")
    axes[1].get_legend().remove()
    
    axes[0].set_ylabel("Density", fontweight = "bold")
    
    old_legend = axes[0].legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    axes[1].legend(handles, labels, ncol = 2, loc='upper right', bbox_to_anchor=(-0.42, -0.13))
    axes[0].get_legend().remove()
    plt.subplots_adjust(wspace = 0.02, hspace = 0.3, top = 0.8)
    
    
    axes[0].set_xlabel("VADER Score", fontweight = "bold", x = 1)
    plt.show()
    
    
    
    
def proportion_of_vader_polarity_over_time():
    us = (pickle.load(open(m4r_data + "us_election_tweets.p", "rb")))[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    ga = (pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb")))[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    
    def f(x):
        if x > 0:
            return 1
        elif x == 0:
            return 0
        else:
            return -1
        
    us_ = us[(us["created_at"] > datetime.datetime(2020, 10, 27, 5, 0)) & (us["created_at"] < datetime.datetime(2020,11,9,5,0))]
        
    us_bot = us_[us_["retweeted_status.id"].isna()] # Drops Retweets
    us_bot = us_bot[us_bot["predicted_class"] == "bot"][["vader", "created_at"]].reset_index(drop = True)
    us_bot["created_at"] = us_bot["created_at"].dt.floor("d")
    us_bot["polarity"] = us_bot["vader"].apply(lambda x: f(x))
    us_bot = us_bot.groupby(["created_at", "polarity"]).count()
    us_bot = us_bot.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    us_bot = us_bot.reset_index()
    
    
    us_human = us_[us_["retweeted_status.id"].isna()] # Drops Retweets
    us_human = us_human[us_human["predicted_class"] == "human"][["vader", "created_at"]].reset_index(drop = True)
    us_human["created_at"] = us_human["created_at"].dt.floor("d")
    us_human["polarity"] = us_human["vader"].apply(lambda x: f(x))
    us_human = us_human.groupby(["created_at", "polarity"]).count()
    us_human = us_human.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    us_human = us_human.reset_index()
    
    pal = [sns.color_palette("tab10")[3], sns.color_palette("tab10")[7], sns.color_palette("tab10")[2]]
    sns.lineplot(data = us_human, x = "created_at", y = "vader", hue = "polarity", palette=pal);
    plt.ylim(19, 55)
    plt.show()
    sns.lineplot(data = us_bot, x = "created_at", y = "vader", hue = "polarity", palette=pal);
    plt.ylim(19, 55)
    plt.show()
    
    
    
    
    
    
def average_vader_sentiment_over_time():
    us = (pickle.load(open(m4r_data + "us_election_tweets.p", "rb")))[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    ga = (pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb")))[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    
    us_ = us[(us["created_at"] > datetime.datetime(2020, 10, 30, 0, 0)) & (us["created_at"] < datetime.datetime(2020,11,8,0,0))]
    ga_ = ga[(ga["created_at"] > datetime.datetime(2021, 1, 1, 0, 0)) & (ga["created_at"] < datetime.datetime(2021,1,10,0,0))]
    

    us_rounded = us_[["vader", "created_at", "predicted_class"]]
    us_rounded["time"] = us_rounded["created_at"].dt.round("6H")
    us_rounded = us_rounded.sort_values("predicted_class", ascending = False)
    
    plt.figure(figsize = (6, 3.6))
    plt.title("Average Tweet Sentiment During the 2020 US Presidential Election", fontweight = "bold")
    plt.xlabel("Date", fontweight = "bold")
    plt.ylabel("Avg. VADER Score over 6 Hours", fontweight = "bold")
    
    plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.09)
    a = sns.lineplot(data = us_rounded, x = "time", y = "vader", hue = "predicted_class")
    #axes[1].axvspan(datetime.datetime(2020, 11, 3, 5, 0), datetime.datetime(2020,11,4,5,0),color = "red", alpha = 0.09)
    
    plt.setp(a.collections[0], alpha=0.1)
    plt.setp(a.collections[1], alpha=0.1)
    
    dtFmt = mdates.DateFormatter('%b %d')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.xticks(fontsize = 9, fontweight = "bold")
    
    
    handles = a.legend_.legendHandles
    labels = ["Human", "Bot"]
    plt.legend(handles, labels, ncol = 2, loc='upper left', bbox_to_anchor=(-0.12, -0.13))
    
    plt.savefig(figure_path + "avg_vader_over_time_us_4_days_either_side.pdf", bbox_inches = "tight")
    
    plt.show()
    
        

    ga_rounded = ga_[["vader", "created_at", "predicted_class"]]
    ga_rounded["time"] = ga_rounded["created_at"].dt.round("6H")
    ga_rounded = ga_rounded.sort_values("predicted_class", ascending = False)
    
    plt.figure(figsize = (6, 3.6))
    plt.title("Average Tweet Sentiment During the 2021 Georgia Runoff Elections", fontweight = "bold")
    plt.xlabel("Date", fontweight = "bold")
    plt.ylabel("Avg. VADER Score over 6 Hours", fontweight = "bold")
    
    plt.axvspan(datetime.datetime(2021, 1, 5, 0, 0), datetime.datetime(2021,1,6,0,0),color = "red", alpha = 0.09)
    a = sns.lineplot(data = ga_rounded, x = "time", y = "vader", hue = "predicted_class")
    #axes[1].axvspan(datetime.datetime(2020, 11, 3, 5, 0), datetime.datetime(2020,11,4,5,0),color = "red", alpha = 0.09)
    
    plt.setp(a.collections[0], alpha=0.1)
    plt.setp(a.collections[1], alpha=0.1)
    
    dtFmt = mdates.DateFormatter('%b %d')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.xticks(fontsize = 9, fontweight = "bold")
    
    
    handles = a.legend_.legendHandles
    labels = ["Human", "Bot"]
    plt.legend(handles, labels, ncol = 2, loc='upper left', bbox_to_anchor=(-0.12, -0.13))
    
    plt.savefig(figure_path + "avg_vader_over_time_ga_4_days_either_side.pdf", bbox_inches = "tight")
    
    plt.show()
    
    