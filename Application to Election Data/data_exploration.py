"""
Title: Data Exploration

Description: Exploring the Training, US Election, and Georgia Election datasets
This includes:
    i. 
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


# 

def comparing_to_training_set():
    df = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    us = df.groupby("user.id").first().reset_index()
    df = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    ga = df.groupby("user.id").first().reset_index()
    trainset = pickle.load(open(m4r_data + "balanced_account_training_data.p", "rb"))
    
    
    follower_data = pd.DataFrame({
        "US" : us["user.followers_count"],
        "Georgia" : ga["user.followers_count"],
        "Training" : trainset["user.followers_count"]
        })
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(
        data = follower_data,
        
        );
    plt.yscale("symlog");
    plt.ylim(0, 1e9);
    plt.title("Comparing User Followers", fontweight = "bold")
    plt.xlabel("Account Dataset", fontweight = "bold")
    plt.ylabel("User Follower Count", fontweight = "bold")
    #plt.savefig(figure_path + "compare_datasets_follower_count.pdf", bbox_inches = "tight")
    plt.show()
    
    
    description_data = pd.DataFrame({
        "US" : us["user.description.length"],
        "Georgia" : ga["user.description.length"],
        "Training" : trainset["user.description.length"]
        })
    
    plt.figure(figsize=(6, 6))
    sns.boxplot(
        data = description_data,
        
        );
    plt.title("Comparing User Description Lengths", fontweight = "bold")
    plt.xlabel("Account Dataset", fontweight = "bold")
    plt.ylabel("User Description Length", fontweight = "bold")
    #plt.savefig(figure_path + "compare_datasets_description_length.pdf", bbox_inches = "tight")
    plt.show()


def proportions_of_bots_and_humans():
    """
    PIE CHARTS:
        - TOTAL # OF ACCOUNTS (Bot : Human)
        - TOTAL # OF TWEETS (Bot : Human)
        - TOTAL # OF REPLIES (Bot : Human)
        - TOTAL # OF RETWEETS (Bot : Human)
    """
    
    df_ga = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    
    userdf = df_ga.groupby("user.id").first()
    retweetdf = df_ga[["retweeted_status.id", "predicted_class"]].dropna()
    replydf = df_ga[["in_reply_to_status_id", "predicted_class"]].dropna()
    
    fig, axes = plt.subplots(1, 4, figsize=(8, 4))
    fig.suptitle('Proportions of Humans and Bots in the US Election Dataset', fontweight = "bold", y = 0.8)
    
    c = ["#83C9F1" , "#FED28F"]
    t = {'fontsize': 15, "fontweight" : "bold"}
    
    # TOTAL # OF USERS PIE CHART
    axes[0].set_title("Accounts", fontweight = "bold");
    h = sum(userdf["predicted_class"] == "human")
    b = sum(userdf["predicted_class"] == "bot")
    counts = [h, b]
    axes[0].pie(counts, autopct='%1.2f%%', colors = c, textprops=t)
    
    # TOTAL # OF TWEETS PIE CHART
    axes[1].set_title("Tweets", fontweight = "bold");
    h = sum(df_ga["predicted_class"] == "human")
    b = sum(df_ga["predicted_class"] == "bot")
    counts = [h, b]
    axes[1].pie(counts, autopct='%1.2f%%', colors = c, textprops=t)
    
    # TOTAL # OF RETWEETS PIE CHART
    axes[2].set_title("Retweets", fontweight = "bold");
    h = sum(retweetdf["predicted_class"] == "human")
    b = sum(retweetdf["predicted_class"] == "bot")
    counts = [h, b]
    axes[2].pie(counts, autopct='%1.2f%%', colors = c, textprops=t)
    
    # TOTAL # OF REPLIES PIE CHART
    axes[3].set_title("Replies", fontweight = "bold");
    h = sum(replydf["predicted_class"] == "human")
    b = sum(replydf["predicted_class"] == "bot")
    counts = [h, b]
    axes[3].pie(counts, autopct='%1.2f%%', colors = c, textprops=t)
    
    plt.subplots_adjust(wspace = 0.05, hspace = -0.5)
    
    handles, labels = axes[0].get_legend_handles_labels()
    
    orange_patch = mpatches.Patch(color=c[1], label='Bot')
    blue_patch = mpatches.Patch(color=c[0], label='Human')

    plt.legend(handles=[blue_patch, orange_patch], bbox_to_anchor=(1, 0.9))
    
    plt.savefig(figure_path + "proportions_pie_chart_us.pdf", bbox_inches = "tight")
    
    plt.show()
    
    
    
    
    
def exploring_user_distribution():
    user_features = ['user.created_at', 'user.name', 'user.screen_name', 'user.description',
       'user.lang', 'user.verified', 'user.geo_enabled',
       'user.default_profile', 'user.default_profile_image',
       'user.followers_count', 'user.friends_count', 'user.listed_count',
       'user.favourites_count', 'user.statuses_count', "predicted_class"]
    us = (pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb")))
    us["tweet_count"] = 1
    us_users = us.groupby("user.id").first()[user_features]
    us_users = us_users.merge(us[["user.id", "tweet_count"]].groupby("user.id").count().reset_index(), how = "left", on = "user.id")
    
    print("There are " + str(len(us_users)) + " unique users")
    
    print("Out of this, there are " + str(sum(us_users["user.verified"])) + " verified users")
    
    print("The top 10% of Users Produced "
          + str( sum(us_users["tweet_count"].sort_values().tail(int(len(us_users) * 0.1))) )
          + " of the tweets in the dataset")
    
    print("The average number of followers the accounts have is " + str(np.mean(us_users["user.followers_count"])))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def compare_vader_score_proportions():
    df_us = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    df_ga = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    
    def f(x):
        if x > 0:
            return 1
        elif x == 0:
            return 0
        else:
            return -1
        
        
    us_bot = df_us[df_us["retweeted_status.id"].isna()] # Drops Retweets
    us_bot = us_bot[us_bot["predicted_class"] == "bot"][["vader", "created_at"]].reset_index(drop = True)
    us_bot["created_at"] = us_bot["created_at"].dt.floor("d")
    us_bot["polarity"] = us_bot["vader"].apply(lambda x: f(x))
    us_bot = us_bot.groupby(["created_at", "polarity"]).count()
    us_bot = us_bot.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    us_bot = us_bot.reset_index()
    
    
    us_human = df_us[df_us["retweeted_status.id"].isna()] # Drops Retweets
    us_human = us_human[us_human["predicted_class"] == "human"][["vader", "created_at"]].reset_index(drop = True)
    us_human["created_at"] = us_human["created_at"].dt.floor("d")
    us_human["polarity"] = us_human["vader"].apply(lambda x: f(x))
    us_human = us_human.groupby(["created_at", "polarity"]).count()
    us_human = us_human.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    us_human = us_human.reset_index()
    
    
    sns.lineplot(data = us_bot, x = "created_at", y = "vader", hue = "polarity");
    sns.lineplot(data = us_human, x = "created_at", y = "vader", hue = "polarity");
    
    
    
    
    
    
    
    
    
    
    us_original = df_us[df_us["retweeted_status.id"].isna() == False]
    us_index = us_original[(us_original["created_at"] > datetime.datetime(2020, 10, 27, 5, 0)) & (us_original["created_at"] < datetime.datetime(2020,11,9,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = us_original.loc[us_index, "vader"]
    plot_1_data["time"] = us_original.loc[us_index, "created_at"].dt.round("D")
    plot_1_data["predicted_class"] = us_original.loc[us_index, "predicted_class"]
    sns.lineplot(data = plot_1_data, x = "time", y = "vader", hue = "predicted_class")
    
    
        
    us_index = df_us[(df_us["created_at"] > datetime.datetime(2020, 10, 27, 5, 0)) & (df_us["created_at"] < datetime.datetime(2020,11,9,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df_us.loc[us_index, "vader"]
    plot_1_data["time"] = df_us.loc[us_index, "created_at"].dt.round("D")
    plot_1_data["predicted_class"] = df_us.loc[us_index, "predicted_class"]
    sns.lineplot(data = plot_1_data, x = "time", y = "vader", hue = "predicted_class")
    
    
    us_pos = df_us[(df_us["vader"] > 0)][["created_at", "vader", "predicted_class"]].reset_index(drop = True)
    us_pos_index = us_pos[(us_pos["created_at"] > datetime.datetime(2020, 10, 27, 5, 0)) & (us_pos["created_at"] < datetime.datetime(2020,11,9,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = us_pos.loc[us_pos_index, "vader"]
    plot_1_data["time"] = us_pos.loc[us_pos_index, "created_at"].dt.round("D")
    plot_1_data["predicted_class"] = us_pos.loc[us_pos_index, "predicted_class"]
    sns.lineplot(data = plot_1_data, x = "time", y = "vader", hue = "predicted_class")
    
    
    us_neg = df_us[(df_us["vader"] < 0)][["created_at", "vader", "predicted_class"]].reset_index(drop = True)
    us_neg_index = us_neg[(us_neg["created_at"] > datetime.datetime(2020, 10, 27, 5, 0)) & (us_neg["created_at"] < datetime.datetime(2020,11,9,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = us_neg.loc[us_neg_index, "vader"]
    plot_1_data["time"] = us_neg.loc[us_neg_index, "created_at"].dt.round("D")
    plot_1_data["predicted_class"] = us_neg.loc[us_neg_index, "predicted_class"]
    sns.lineplot(data = plot_1_data, x = "time", y = "vader", hue = "predicted_class")
    
    # zoomed_index = df_us[(df_us["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (df_us["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    
    
    
    
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex = True)
    
    us_neg_hum = df_us[(df_us["vader"] < 0) & (df_us["predicted_class"] == "human")][["created_at", "vader", "predicted_class"]].reset_index(drop = True)
    us_neg_hum_index = us_neg_hum[(us_neg_hum["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (us_neg_hum["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = us_neg_hum.loc[us_neg_hum_index, "vader"]
    plot_1_data["time"] = us_neg_hum.loc[us_neg_hum_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = us_neg_hum.loc[us_neg_hum_index, "predicted_class"]
    sns.lineplot(ax = axes[0], data = plot_1_data, x = "time", y = "vader")
    axes[0].set_ylim(-0.6, -0.4)
    
    us_neg_bot = df_us[(df_us["vader"] < 0) & (df_us["predicted_class"] == "bot")][["created_at", "vader", "predicted_class"]].reset_index(drop = True)
    us_neg_bot_index = us_neg_bot[(us_neg_bot["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (us_neg_bot["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = us_neg_bot.loc[us_neg_bot_index, "vader"]
    plot_1_data["time"] = us_neg_bot.loc[us_neg_bot_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = us_neg_bot.loc[us_neg_bot_index, "predicted_class"]
    sns.lineplot(ax = axes[1], data = plot_1_data, x = "time", y = "vader")
    axes[1].set_ylim(-0.6, -0.4)
    
    
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 5.5), sharex = True)
    
    us_pos_hum = df_us[(df_us["vader"] > 0) & (df_us["predicted_class"] == "human")][["created_at", "vader", "predicted_class"]].reset_index(drop = True)
    us_pos_hum_index = us_pos_hum[(us_pos_hum["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (us_pos_hum["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = us_pos_hum.loc[us_pos_hum_index, "vader"]
    plot_1_data["time"] = us_pos_hum.loc[us_pos_hum_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = us_pos_hum.loc[us_pos_hum_index, "predicted_class"]
    sns.lineplot(ax = axes[0], data = plot_1_data, x = "time", y = "vader")
    
    us_pos_bot = df_us[(df_us["vader"] > 0) & (df_us["predicted_class"] == "bot")][["created_at", "vader", "predicted_class"]].reset_index(drop = True)
    us_pos_bot_index = us_pos_bot[(us_pos_bot["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (us_pos_bot["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = us_pos_bot.loc[us_pos_bot_index, "vader"]
    plot_1_data["time"] = us_pos_bot.loc[us_pos_bot_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = us_pos_bot.loc[us_pos_bot_index, "predicted_class"]
    sns.lineplot(ax = axes[1], data = plot_1_data, x = "time", y = "vader")
    
    
    
    fig.suptitle('VADER Polarity Scores Across Accounts With a 5 Tweet Corpus', fontweight = "bold")
    
    axes[0].set_ylabel("Density", fontweight = "bold")
    axes[1].set_xlabel("VADER Polarity Score", fontweight = "bold")
    
    
    
    
    
    
    sns.lineplot(data = df_us[df_us["vader"] > 0], x = "created_at", y = "vader", hue = "predicted_class")
    
    
    sns.histplot(
        ax = axes[0],
        data = DF,#[DF["vader"] != 0],
        x = "vader",
        hue = "class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[0].set_title("Training Dataset", fontweight = "bold");
    axes[0].set_xlabel("")
    
    old_legend = axes[0].legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "Class"
    axes[0].legend(handles, labels, title=title, loc='upper left')
    
    sns.histplot(
        ax = axes[1],
        data = df_us[df_us["vader"] > 0],#[DF_us["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[1].set_title("US Election Dataset", fontweight = "bold");
    axes[1].get_legend().remove()
    
    sns.histplot(
        ax = axes[2],
        data = df_ga,#[DF_ga["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[2].set_title("Georgia Election Dataset", fontweight = "bold");
    axes[2].get_legend().remove()
    axes[2].set_xlabel("")
    
    
    plt.subplots_adjust(wspace = 0.05, hspace = 0.1)
    #fig.tight_layout(pad = 0.1)
    #plt.savefig(figure_path + "compare_vadersdlkfjhkahfd_polarity_scores_5_tweets_concat.pdf", bbox_inches = "tight")
    plt.show()





def applying_account_level_detection():
    df_us = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    us = df_us.groupby("user.id").first().reset_index()
    df_ga = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    ga = df_ga.groupby("user.id").first().reset_index()
    
    
    #trainset = pickle.load(open(m4r_data + "balanced_account_training_data.p", "rb"))
    
    
    trainset = get_full_dataset()
    us = get_full_dataset(us)
    ga = get_full_dataset(ga)
    
    X_us = us[features]
    X_ga = ga[features]
    
    X_trn = trainset[features]
    y_trn = trainset["class"].replace({"bot" : 1, "human" : 0})
                                      
    SME = SMOTEENN(random_state = 2727841)
    X_trn, y_trn = SME.fit_resample(X_trn, y_trn)
    
    scaling = StandardScaler()
    X_trn = scaling.fit_transform(X_trn)
    X_us  = scaling.transform(X_us)
    X_ga  = scaling.transform(X_ga)

    clf = AdaBoostClassifier(n_estimators = 50, random_state = 9926737)
    clf.fit(X_trn, y_trn)    
    
    p_us = np.round(clf.predict(X_us))
    p_ga = np.round(clf.predict(X_ga))
    
    us["predicted_class"] = p_us
    ga["predicted_class"] = p_ga
    
    us["predicted_class"] = us["predicted_class"].replace({0 : "human", 1 : "bot"})
    ga["predicted_class"] = ga["predicted_class"].replace({0 : "human", 1 : "bot"})
    
    print("% Bots (US): ", sum(p_us)/len(p_us))
    print("% Bots (GA): ", sum(p_ga)/len(p_ga))
    
    # adding predicted class column to df
    df_us = df_us.merge(us[["user.id", "predicted_class"]], how = "left", on = "user.id")
    # adding predicted class column to df
    df_ga = df_ga.merge(ga[["user.id", "predicted_class"]], how = "left", on = "user.id")
    
    
    
    ########
    # PLOT 1
    ########
    plt.figure(figsize=(8, 8))
    ax = sns.histplot(
        data = df_us[df_us["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21)
        )
    ax.set_title("VADER Polarity Scores for US Election Tweets with Neutral Tweets Removed", fontweight = "bold");
    ax.set_xlabel("VADER Polarity Score", fontweight = "bold")
    ax.set_ylabel("Density", fontweight = "bold")
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "(Predicted)  Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    plt.savefig(figure_path + "compare_vader_polarity_scores_US_neutral_removed.pdf", bbox_inches = "tight")
    plt.show()
    
    ########
    # PLOT 2
    ########
    plt.figure(figsize=(8, 8))
    ax = sns.histplot(
        data = df_ga[df_ga["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21)
        )
    ax.set_title("VADER Polarity Scores for Georgia Election Tweets with Neutral Tweets Removed", fontweight = "bold");
    ax.set_xlabel("VADER Polarity Score", fontweight = "bold")
    ax.set_ylabel("Density", fontweight = "bold")
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "(Predicted) Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    plt.savefig(figure_path + "compare_vader_polarity_scores_GA_neutral_removed.pdf", bbox_inches = "tight")
    plt.show()
    
    
    ########
    # PLOT 3
    ########
    # Now applying concatenation of 5 Tweet Corpus...
    # CONCATENATING TWEETS
    N_TWEETS = 5
    
    # finding number of users with at least 5 tweets...
    print("Number of users with at least " + str(N_TWEETS) + " tweets: ", sum(df_us.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    # picking out user ids with more than N_TWEETS tweets
    available_users = (df_us.groupby(["user.id"]).first()[df_us.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    # joining the firt N_TWEETS tweets together to form a corpus for the dataset
    DF_us = (df_us[df_us["user.id"].isin(available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    # Applying Vader analysis to it (doesn't take too long)
    sid = SentimentIntensityAnalyzer()
    DF_us["vader"] = DF_us["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    DF_us = DF_us.merge(df_us.groupby(["user.id"]).first().reset_index()[["user.id", "predicted_class"]], how = "left", on = "user.id")
    # Plotting:
    plt.figure(figsize=(8, 8))
    ax = sns.histplot(
        data = DF_us,#[DF_us["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    ax.set_title("VADER Polarity Scores for US Election Tweets with 5 Tweet Corpus", fontweight = "bold");
    ax.set_xlabel("VADER Polarity Score", fontweight = "bold")
    ax.set_ylabel("Density", fontweight = "bold")
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "(Predicted) Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    plt.savefig(figure_path + "compare_vader_polarity_scores_us_5_tweets_concat.pdf", bbox_inches = "tight")
    plt.show()
    
    
    # NOW THE SAME FOR GEORGIA
    # CONCATENATING TWEETS
    N_TWEETS = 5
    
    # finding number of users with at least 5 tweets...
    print("Number of users with at least " + str(N_TWEETS) + " tweets: ", sum(df_ga.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    # picking out user ids with more than N_TWEETS tweets
    available_users = (df_ga.groupby(["user.id"]).first()[df_ga.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    # joining the firt N_TWEETS tweets together to form a corpus for the dataset
    DF_ga = (df_ga[df_ga["user.id"].isin(available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    # Applying Vader analysis to it (doesn't take too long)
    sid = SentimentIntensityAnalyzer()
    DF_ga["vader"] = DF_ga["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    DF_ga = DF_ga.merge(df_ga.groupby(["user.id"]).first().reset_index()[["user.id", "predicted_class"]], how = "left", on = "user.id")
    # Plotting:
    plt.figure(figsize=(8, 8))
    ax = sns.histplot(
        data = DF_ga,#[DF["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    ax.set_title("VADER Polarity Scores for Georgia Election Tweets with 5 Tweet Corpus", fontweight = "bold");
    ax.set_xlabel("VADER Polarity Score", fontweight = "bold")
    ax.set_ylabel("Density", fontweight = "bold")
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "(Predicted) Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    plt.savefig(figure_path + "compare_vader_polarity_scores_ga_5_tweets_concat.pdf", bbox_inches = "tight")
    plt.show()
    
    # NOW THE SAME FOR TRAINING DATA
    # CONCATENATING TWEETS
    df = pickle.load(open(m4r_data + "balanced_tweet_training_data.p", "rb"))
    # df["vader"]  = df["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    N_TWEETS = 5
    print("Number of users with at least " + str(N_TWEETS) + " tweets: ", sum(df.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    # picking out user ids with more than N_TWEETS tweets
    available_users = (df.groupby(["user.id"]).first()[df.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    # joining the firt N_TWEETS tweets together to form a corpus for the dataset
    DF = (df[df["user.id"].isin(available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    # Applying Vader analysis to it (doesn't take too long)
    sid = SentimentIntensityAnalyzer()
    DF["vader"] = DF["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    DF = DF.merge(df.groupby(["user.id"]).first().reset_index()[["user.id", "class"]], how = "left", on = "user.id")
    
    
    ######################################################
    # SIDE BY SIDE PLOT
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey = True)
    fig.suptitle('VADER Polarity Scores Across Accounts With a 5 Tweet Corpus', fontweight = "bold")
    
    axes[0].set_ylabel("Density", fontweight = "bold")
    axes[1].set_xlabel("VADER Polarity Score", fontweight = "bold")
    
    sns.histplot(
        ax = axes[0],
        data = DF,#[DF["vader"] != 0],
        x = "vader",
        hue = "class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[0].set_title("Training Dataset", fontweight = "bold");
    axes[0].set_xlabel("")
    
    old_legend = axes[0].legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "Class"
    axes[0].legend(handles, labels, title=title, loc='upper left')
    
    sns.histplot(
        ax = axes[1],
        data = DF_us,#[DF_us["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[1].set_title("US Election Dataset", fontweight = "bold");
    axes[1].get_legend().remove()
    
    sns.histplot(
        ax = axes[2],
        data = DF_ga,#[DF_ga["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[2].set_title("Georgia Election Dataset", fontweight = "bold");
    axes[2].get_legend().remove()
    axes[2].set_xlabel("")
    
    
    plt.subplots_adjust(wspace = 0.05, hspace = 0.1)
    #fig.tight_layout(pad = 0.1)
    plt.savefig(figure_path + "compare_vader_polarity_scores_5_tweets_concat.pdf", bbox_inches = "tight")
    plt.show()
    
    #################################################
    # NOW APPLYING VADER TO TRAINING DATASET
    #sid = SentimentIntensityAnalyzer()
    #df["vader"] = df["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    ######################################################
    # SIDE BY SIDE PLOT
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey = True)
    fig.suptitle('VADER Polarity Scores Across Tweets With Neutral Scores Removed', fontweight = "bold")
    
    axes[0].set_ylabel("Density", fontweight = "bold")
    axes[1].set_xlabel("VADER Polarity Score", fontweight = "bold")
    
    sns.histplot(
        ax = axes[0],
        data = df[df["vader"] != 0],
        x = "vader",
        hue = "class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[0].set_title("Training Dataset", fontweight = "bold");
    axes[0].set_xlabel("")
    
    old_legend = axes[0].legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "Class"
    axes[0].legend(handles, labels, title=title, loc='upper left')
    
    sns.histplot(
        ax = axes[1],
        data = df_us[df_us["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[1].set_title("US Election Dataset", fontweight = "bold");
    axes[1].get_legend().remove()
    
    sns.histplot(
        ax = axes[2],
        data = df_ga[df_ga["vader"] != 0],
        x = "vader",
        hue = "predicted_class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    axes[2].set_title("Georgia Election Dataset", fontweight = "bold");
    axes[2].get_legend().remove()
    axes[2].set_xlabel("")
    
    
    plt.subplots_adjust(wspace = 0.05, hspace = 0.1)
    #fig.tight_layout(pad = 0.1)
    plt.savefig(figure_path + "compare_vader_polarity_scores_neutral_removed.pdf", bbox_inches = "tight")
    plt.show()
    
    
    
    
def plot_us_election_vader_day_of_election():
    df_us = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    
    
    
    
    
    
    
    
def plot_vader_score_over_time_day_of(df_us, df_ga):
    # Assume df_us and df_ga already have been classified and have a predicted_class column!
    df = df_us[df_us["vader"] != 0];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (df["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_1_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    ax = sns.lineplot(data = plot_1_data, x = "time", y = "vader", hue = "predicted_class")
    ax.set_title("Average VADER Polarity Scores for US Election Dataset On November 3rd 2020", fontweight = "bold")
    ax.set_xlabel("Date (UTC+0000)", fontweight = "bold")
    ax.set_ylabel("Average VADER Polarity Score (Aggregated by Hour)", fontweight = "bold")
    #plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.03)
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    
    plt.show()
    
    
    df = df_ga[df_ga["vader"] != 0];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2021, 1, 5, 5, 0)) & (df["created_at"] < datetime.datetime(2021, 1, 6,5,0))].index
    plot_2_data = pd.DataFrame()
    plot_2_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_2_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("H")
    plot_2_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    ax = sns.lineplot(data = plot_2_data, x = "time", y = "vader", hue = "predicted_class")
    ax.set_title("Average VADER Polarity Scores for Georgia Election Dataset On January 5th 2021", fontweight = "bold")
    ax.set_xlabel("Date (UTC+0000)", fontweight = "bold")
    ax.set_ylabel("Average VADER Polarity Score (Aggregated by Hour)", fontweight = "bold")
    #plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.03)
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    
    plt.show()
    
def plot_vader_score_over_time_DAY_OF(df_us, df_ga):
    # Assume df_us and df_ga already have been classified and have a predicted_class column!
    df = df_us[(df_us["vader"] != 0) & (df_us["predicted_class"] == "human")];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (df["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_1_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    ##
    df = df_us[(df_us["vader"] != 0) & (df_us["predicted_class"] == "bot")];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 11, 3, 5, 0)) & (df["created_at"] < datetime.datetime(2020,11,4,5,0))].index
    plot_2_data = pd.DataFrame()
    plot_2_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_2_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("H")
    plot_2_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    
    # PLOT ON TOP OF EACH OTHER
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex = True, sharey = True)
    fig.suptitle('Average VADER Polarity Scores Over Time For US Election Tweets', fontweight = "bold")
    
    axes[0].set_ylabel("Avg. Polarity Score", fontweight = "bold")
    axes[1].set_ylabel("Avg. Polarity Score", fontweight = "bold")
    axes[1].set_xlabel("Date (UTC+0000)", fontweight = "bold")
    
    sns.lineplot(ax = axes[0], data = plot_1_data, x = "time", y = "vader")
    axes[0].set_title("Human Tweets", fontweight = "bold")
    axes[0].set_xlabel("")
    #plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.03)
    
    sns.lineplot(ax = axes[1], data = plot_2_data, x = "time", y = "vader")
    axes[1].set_title("Bot Tweets", fontweight = "bold")
    #plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.03)
    
    plt.subplots_adjust(wspace = 0.05, hspace = 0.13)
    plt.savefig(figure_path + "avg_vader_over_time_day_of_neu_removed_us.pdf", bbox_inches = "tight")
    plt.show()
    
def plot_vader_score_over_time_DAYS_AROUND(df_us, df_ga):
    # Assume df_us and df_ga already have been classified and have a predicted_class column!
    df = df_us[(df_us["vader"] != 0) & (df_us["predicted_class"] == "human")];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 10, 25, 5, 0)) & (df["created_at"] < datetime.datetime(2020,11,14,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_1_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("6H")
    plot_1_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    ##
    df = df_us[(df_us["vader"] != 0) & (df_us["predicted_class"] == "bot")];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 10, 25, 5, 0)) & (df["created_at"] < datetime.datetime(2020,11,14,5,0))].index
    plot_2_data = pd.DataFrame()
    plot_2_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_2_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("6H")
    plot_2_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    
    # PLOT ON TOP OF EACH OTHER
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex = True, sharey = True)
    fig.suptitle('Average VADER Polarity Scores Over Time For US Election Tweets', fontweight = "bold")
    
    axes[0].set_ylabel("Avg. Polarity Score", fontweight = "bold")
    axes[1].set_ylabel("Avg. Polarity Score", fontweight = "bold")
    axes[1].set_xlabel("Date (UTC+0000)", fontweight = "bold")
    
    sns.lineplot(ax = axes[0], data = plot_1_data, x = "time", y = "vader")
    axes[0].set_title("Human Tweets", fontweight = "bold")
    axes[0].set_xlabel("")
    axes[0].axvspan(datetime.datetime(2020, 11, 3, 5, 0), datetime.datetime(2020,11,4,5,0),color = "red", alpha = 0.09)
    
    sns.lineplot(ax = axes[1], data = plot_2_data, x = "time", y = "vader")
    axes[1].set_title("Bot Tweets", fontweight = "bold")
    axes[1].axvspan(datetime.datetime(2020, 11, 3, 5, 0), datetime.datetime(2020,11,4,5,0),color = "red", alpha = 0.09)
    
    plt.subplots_adjust(wspace = 0.05, hspace = 0.13)
    
    plt.savefig(figure_path + "avg_vader_over_time_days_around_neu_removed_us.pdf", bbox_inches = "tight")
    plt.show()
    
    
    dtFmt = mdates.DateFormatter('%b %d\n%Y') # define the formatting
    
    
    
    # GEORGIA
    
    # Assume df_us and df_ga already have been classified and have a predicted_class column!
    df = df_ga[(df_ga["vader"] != 0) & (df_ga["predicted_class"] == "human")];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 12, 20, 5, 0)) & (df["created_at"] < datetime.datetime(2021,1,9,5,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_1_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("6H")
    plot_1_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    ##
    df = df_ga[(df_ga["vader"] != 0) & (df_ga["predicted_class"] == "bot")];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 12, 20, 5, 0)) & (df["created_at"] < datetime.datetime(2021,1,9,5,0))].index
    plot_2_data = pd.DataFrame()
    plot_2_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_2_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("6H")
    plot_2_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    
    # PLOT ON TOP OF EACH OTHER
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex = True, sharey = True)
    fig.suptitle('Average VADER Polarity Scores Over Time For Georgia Election Tweets', fontweight = "bold")
    
    axes[0].set_ylabel("Avg. Polarity Score", fontweight = "bold")
    axes[1].set_ylabel("Avg. Polarity Score", fontweight = "bold")
    axes[1].set_xlabel("Date (UTC+0000)", fontweight = "bold")
    
    sns.lineplot(ax = axes[0], data = plot_1_data, x = "time", y = "vader")
    axes[0].set_title("Human Tweets", fontweight = "bold")
    axes[0].set_xlabel("")
    axes[0].axvspan(datetime.datetime(2021, 1, 5, 5, 0), datetime.datetime(2021,1,6,5,0),color = "red", alpha = 0.09)
    
    sns.lineplot(ax = axes[1], data = plot_2_data, x = "time", y = "vader")
    axes[1].set_title("Bot Tweets", fontweight = "bold")
    axes[1].axvspan(datetime.datetime(2021, 1, 5, 5, 0), datetime.datetime(2021,1,6,5,0),color = "red", alpha = 0.09)
    plt.gca().xaxis.set_major_formatter(dtFmt)
    
    plt.subplots_adjust(wspace = 0.05, hspace = 0.13)
    
    plt.savefig(figure_path + "avg_vader_over_time_days_around_neu_removed_ga.pdf", bbox_inches = "tight")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def plot_vader_score_over_time(us, ga):
    df = us[us["vader"] != 0];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 11, 1, 0, 0)) & (df["created_at"] < datetime.datetime(2020,11,6,0,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_1_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    ax = sns.lineplot(data = plot_1_data, x = "time", y = "vader", hue = "predicted_class")
    ax.set_title("Average Compound Vader Scores between Nov 1st to Nov 6th 2020")
    ax.set(xlabel='Date', ylabel = "Average of Compound Vader Score (Aggregated by Hours)")
    plt.axvspan(datetime.datetime(2020, 11, 3, 0, 0), datetime.datetime(2020,11,4,0,0),color = "red", alpha = 0.03)
    plt.show()
    
    df = ga[ga["vader"] != 0];
    zoomed_index = df[(df["created_at"] > datetime.datetime(2021, 1, 3, 0, 0)) & (df["created_at"] < datetime.datetime(2021, 1, 8, 0, 0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_1_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("H")
    plot_1_data["predicted_class"] = df.loc[zoomed_index, "predicted_class"]
    ax = sns.lineplot(data = plot_1_data, x = "time", y = "vader", hue = "predicted_class")
    ax.set_title("Average Compound Vader Scores between Jan 3rd to Jan 8th 2020")
    ax.set(xlabel='Date', ylabel = "Average of Compound Vader Score (Aggregated by Hours)")
    plt.axvspan(datetime.datetime(2021, 1, 5, 0, 0), datetime.datetime(2021, 1, 6, 0, 0),color = "red", alpha = 0.03)
    plt.show()
    

def vader_score_over_time(df):
    """
    Plotting the Vader score vs time
    """
    # Zoomed in
    zoomed_index = df[(df["created_at"] > datetime.datetime(2020, 11, 1, 0, 0)) & (df["created_at"] < datetime.datetime(2020,11,6,0,0))].index
    plot_1_data = pd.DataFrame()
    plot_1_data["vader"] = df.loc[zoomed_index, "vader"]
    plot_1_data["time"] = df.loc[zoomed_index, "created_at"].dt.round("H")
    ax = sns.lineplot(data = plot_1_data, x = "time", y = "vader")
    ax.set_title("Average Compound Vader Scores between Nov 1st to Nov 6th 2020")
    ax.set(xlabel='Date', ylabel = "Average of Compound Vader Score (Aggregated by Hours)")
    plt.show()
    
    # Zoomed in and neutral scores removed
    zoomed_index_2 = df[((df["created_at"] > datetime.datetime(2020, 11, 1, 0, 0)) & (df["created_at"] < datetime.datetime(2020,11,6,0,0))) & (df["vader"] != 0)].index
    plot_2_data = pd.DataFrame()
    plot_2_data["vader"] = df.loc[zoomed_index_2, "vader"]
    plot_2_data["time"] = df.loc[zoomed_index_2, "created_at"].dt.round("H")
    ax = sns.lineplot(data = plot_2_data, x = "time", y = "vader")
    ax.set_title("Average Compound Vader Scores between Nov 1st to Nov 6th 2020, Neutral Scores removed")
    ax.set(xlabel='Date', ylabel = "Average of Compound Vader Score (Aggregated by Hours)")
    plt.show()
    
    # Not zoomed in:
    unzoomed_index = df[df["created_at"] > datetime.datetime(2020, 10, 23, 0, 0)].index
    plot_3_data = pd.DataFrame()
    plot_3_data["vader"] = df.loc[unzoomed_index, "vader"]
    plot_3_data["time"] = df.loc[unzoomed_index, "created_at"].dt.round("0.5D")
    ax = sns.lineplot(data = plot_3_data, x = "time", y = "vader")
    ax.set_title("Average Compound Vader Scores between Oct 23rd to Nov 14th 2020")
    ax.set(xlabel='Date', ylabel = "Average of Compound Vader Score (Aggregated by Half Days)")
    plt.show()




def checking_distribution_of_vader_scores_in_training_tweets():
    df = pickle.load(open(m4r_data + "balanced_tweet_training_data.p", "rb"))
    sid = SentimentIntensityAnalyzer()
    df["vader"]  = df["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    # (takes about 9 minutes)
    
    
    
    ######
    # PLOT 
    ######
    
    plt.figure(figsize=(8, 8))
    ax = sns.histplot(
        data = df[df["vader"] != 0],
        x = "vader",
        hue = "class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    ax.set_title("VADER Polarity Scores for Training Dataset Tweets with Neutral Tweets Removed", fontweight = "bold");
    ax.set_xlabel("VADER Polarity Score", fontweight = "bold")
    ax.set_ylabel("Density", fontweight = "bold")
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    plt.savefig(figure_path + "compare_vader_polarity_scores_training_neutral_removed.pdf", bbox_inches = "tight")
    plt.show()
    
    # CONCATENATING TWEETS
    N_TWEETS = 5
    print("Number of users with at least " + str(N_TWEETS) + " tweets: ", sum(df.groupby(["user.id"])["user.id"].count() >= N_TWEETS))
    # picking out user ids with more than N_TWEETS tweets
    available_users = (df.groupby(["user.id"]).first()[df.groupby(["user.id"])["user.id"].count() >= N_TWEETS]).reset_index()["user.id"]
    # joining the firt N_TWEETS tweets together to form a corpus for the dataset
    DF = (df[df["user.id"].isin(available_users)].groupby(["user.id"]).head(N_TWEETS)[["user.id", "full_text"]]).groupby(["user.id"])["full_text"].apply(" ".join).reset_index()
    # Applying Vader analysis to it (doesn't take too long)
    sid = SentimentIntensityAnalyzer()
    DF["vader"] = DF["full_text"].apply(lambda x : sid.polarity_scores(x)["compound"])
    DF = DF.merge(df.groupby(["user.id"]).first().reset_index()[["user.id", "class"]], how = "left", on = "user.id")
    # Plotting:
    plt.figure(figsize=(8, 8))
    ax = sns.histplot(
        data = DF,#[DF["vader"] != 0],
        x = "vader",
        hue = "class",
        stat="density",
        common_norm = False,
        bins = np.linspace(-1,1,21),
        element = "step"
        )
    ax.set_title("VADER Polarity Scores for Training Dataset Tweets with 5 Tweet Corpus", fontweight = "bold");
    ax.set_xlabel("VADER Polarity Score", fontweight = "bold")
    ax.set_ylabel("Density", fontweight = "bold")
    
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = ["Human", "Bot"] # [t.get_text() for t in old_legend.get_texts()]
    title = "Class"
    ax.legend(handles, labels, title=title, loc='upper left')
    plt.savefig(figure_path + "compare_vader_polarity_scores_training_5_tweets_concat.pdf", bbox_inches = "tight")
    plt.show()



def concatenating_tweets():
    pass










# sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection")
# from tweet_level_detection import contextual_lstm_model
# from tweet_level_detection_preprocessing_pt_3 import load_sample, tensorfy, create_embeddings_index, create_embedding_matrix



    
    
    
def cross_validation_of_classifiers(df):
    weightsavepath = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection\\contextual_LSTM_weights\\"
    
    df_trn = load_sample(5000000)
    vectorizer, scaling, trn_tweet_vector, trn_metadata, trn_labels = tensorfy(df_trn)
    embeddings_index = create_embeddings_index()
    embedding_matrix = create_embedding_matrix(vectorizer, embeddings_index)
    optimizer = "Adam"
    model = contextual_lstm_model(embedding_matrix, optimizer)
    model.load_weights(weightsavepath)
    
    
    # Vectorizing new input data in batches
    vectorizer_batch_size = 1024
    vectorized_new_data_batches = []
    r = int(np.ceil(len(df)/vectorizer_batch_size))
    print(" ")
    for i in range(r):
        sys.stdout.write("\r" + str(i+1) + " out of " + str(r))
        vectorized_new_data_batches.append(vectorizer(df["tokenised_text"][vectorizer_batch_size*i:vectorizer_batch_size*(i+1)].tolist()))
    tweet_vectors = tf.concat(vectorized_new_data_batches, axis = 0)
    print("                           ")
    print("Finished Vectorizing")
    
    tweet_metadata = df[['hashtag_count', 'mention_count', 'url_count',
       'retweet_count', 'favorite_count']].to_numpy()
    tweet_metadata = scaling.transform(tweet_metadata)
    
    
    contextual_lstm_predictions = model.predict({"tweet_input" : tweet_vectors, "metadata_input" : tweet_metadata})
    
    y = np.array(contextual_lstm_predictions).reshape(2, -1)
    
    y_final = y[0,:]
    
    y_final.round().astype(int)
    
    # NOTE: THE FIRST COLUMN IS THE final output, SECOND IS AUXILIARY OUTPUT!
    
    # checking difference between auxiliary and final output:
    
    sum((y_final.round().astype(int) - y[1,:].round().astype(int)) != 0)
    
    # pickle.dump(y_final.round().astype(int), open(collected_data + "Predictions\\contextual_lstm_predictions_2020-03-21.p", "wb"))
    
    
    plt.hist(y[0,:]); plt.title("Histogram plot of Contextual LSTM probabilities for US Election data"); plt.ylabel("Count"); plt.xlabel("Final Output Probability"); plt.show()
    
    