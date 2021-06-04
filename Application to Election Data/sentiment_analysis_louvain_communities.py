"""
Title: Sentiment Analysis Applied to the Louvain Communities found for the Georgia Reply Network

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
    ga = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    gephi = pd.read_csv(m4r_data + "Georgia Reply Network Louvain Community Detection.csv")
    gephi.columns = ["user.id", "label", "timeset", "Community"]
    gephi = gephi[gephi["Community"].isin([8, 42])]
    
    ga_ = ga[ga["user.id"].isin(gephi["user.id"])]
    ga_ = ga_.merge(gephi[["user.id", "Community"]], how = "left", on = "user.id")
    # CONCATENATING TWEETS
    N_TWEETS = 5
    print("users with at least " + str(N_TWEETS) + " tweets: ", sum(ga_.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    ga_available_users = (ga_.groupby(["user.id"]).first()[ga_.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    # joining the firt N_TWEETS tweets together to form a corpus for the dataset
    ga_2 = (ga_[ga_["user.id"].isin(ga_available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    sid = SentimentIntensityAnalyzer()
    ga_2["vader"] = ga_2["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    ga_2 = ga_2.merge(ga_.groupby(["user.id"]).first().reset_index()[["user.id", "predicted_class"]], how = "left", on = "user.id")
    ga_2 = ga_2.merge(gephi[["user.id", "Community"]], how = "left", on = "user.id")
    
    
        
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.9), sharey = True)
    fig.suptitle('Distribution of VADER Scores', fontweight = "bold")
    # 1.
    sns.histplot(ax = axes[0],
                 data = ga_.sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "Community",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step",
        )
    axes[0].set_title("No Adjustment", fontweight = "bold");
    axes[0].set_xlabel("")
    # 2.
    sns.histplot(ax = axes[1],
                 data = ga_[ga_["vader"] != 0].sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "Community",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step",
        )
    axes[1].set_title("Neutral Removed", fontweight = "bold");
    axes[1].set_xlabel("")
    # 3.
    sns.histplot(ax = axes[2],
                 data = ga_2.sort_values("predicted_class", ascending = False),
                 x = "vader",
                 hue = "Community",
                 stat="density",
                 common_norm = False,
                 bins = np.linspace(-1,1,20),
                 element = "step",
        )
    axes[2].set_title("No Adjustment", fontweight = "bold");
    axes[2].set_xlabel("")
    
    plt.show()

    
    
    
    
def average_vader_sentiment_over_time():
    
    ga = (pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb")))[["id", "user.id", "created_at", "full_text", "tokenised_text", "user.verified", "in_reply_to_status_id", "retweeted_status.id", 'vader', 'predicted_class']]
    gephi = pd.read_csv(m4r_data + "Georgia Reply Network Louvain Community Detection.csv")
    gephi.columns = ["user.id", "label", "timeset", "Community"]
    gephi = gephi[gephi["Community"].isin([8, 42])]
    ga_ = ga[ga["user.id"].isin(gephi["user.id"])]
    ga_ = ga_.merge(gephi[["user.id", "Community"]], how = "left", on = "user.id")
    ga_ = ga_[(ga_["created_at"] > datetime.datetime(2021, 1, 1, 0, 0)) & (ga_["created_at"] < datetime.datetime(2021,1,9,0,0))]
    

    ga__rounded = ga_[["vader", "created_at", "predicted_class", "Community"]]
    ga__rounded["time"] = ga__rounded["created_at"].dt.round("6H")
    ga__rounded = ga__rounded.sort_values("Community", ascending = False)
    
    plt.figure(figsize = (6, 3.6))
    plt.title("Average VADER Sentiment Around the 2021 Georgia Election Day", fontweight = "bold")
    plt.xlabel("Date", fontweight = "bold")
    plt.ylabel("Avg. VADER Score over 6 Hours", fontweight = "bold")
    
    plt.axvspan(datetime.datetime(2021, 1, 5, 0, 0), datetime.datetime(2021,1,6,0,0),color = "red", alpha = 0.09)
    a = sns.lineplot(data = ga__rounded, x = "time", y = "vader", hue = "Community", palette = [sns.color_palette("hls", 8)[0], sns.color_palette("hls", 8)[5]])
    #axes[1].axvspan(datetime.datetime(2020, 11, 3, 5, 0), datetime.datetime(2020,11,4,5,0),color = "red", alpha = 0.09)
    
    plt.setp(a.collections[0], alpha=0.1)
    plt.setp(a.collections[1], alpha=0.1)
    
    dtFmt = mdates.DateFormatter('%b %d')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(dtFmt)
    plt.xticks(fontsize = 9, fontweight = "bold")
    
    
    handles = a.legend_.legendHandles
    labels = ["Rep (42)", "Dem (8)"]
    plt.legend(handles, labels, ncol = 2, loc='upper left', bbox_to_anchor=(-0.12, -0.13))
    
    
    plt.show()
    
    
    
    
    
    fig, axes = plt.subplots(1,2, figsize = (7, 3.6), sharey = True)
    fig.suptitle("Average VADER Sentiment Around the 2021 Georgia Election Day", fontweight = "bold")
    axes[0].set_xlabel("Date", fontweight = "bold")
    axes[0].set_ylabel("Avg. VADER Score over 6 Hours", fontweight = "bold")
    axes[0].set_title("Community 8", fontweight = "bold")
    axes[0].axvspan(datetime.datetime(2021, 1, 5, 0, 0), datetime.datetime(2021,1,6,0,0),color = "red", alpha = 0.09)
    a = sns.lineplot(ax = axes[0], data = ga__rounded[ga__rounded["Community"] == 8], x = "time", y = "vader", hue = "predicted_class")
    
    plt.setp(a.collections[0], alpha=0.1)
    plt.setp(a.collections[1], alpha=0.1)
    
    handles = a.legend_.legendHandles
    labels = ["Rep (42)", "Dem (8)"]
    plt.legend(handles, labels, ncol = 2, loc='upper left', bbox_to_anchor=(-0.12, -0.13))
    
    axes[1].set_xlabel("", fontweight = "bold")
    axes[1].set_title("Community 42", fontweight = "bold")
    axes[1].axvspan(datetime.datetime(2021, 1, 5, 0, 0), datetime.datetime(2021,1,6,0,0),color = "red", alpha = 0.09)
    b = sns.lineplot(ax = axes[1], data = ga__rounded[ga__rounded["Community"] == 42], x = "time", y = "vader", hue = "predicted_class",)
    
    plt.setp(b.collections[0], alpha=0.1)
    plt.setp(b.collections[1], alpha=0.1)
    
    plt.show()
    
    
    