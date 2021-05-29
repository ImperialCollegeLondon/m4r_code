# -*- coding: utf-8 -*-
"""
Feature Selection:
Principal Components Analysis & RFC Important Features
"""

# 1. SETUP --------------------------------------------------------------------
import pickle, math, sys, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# Setting the plot styles etc.
sns.set(font="Arial")
# LOAD ACCOUNT LEVEL DETECTION:
#sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_repository\\")
#from account_level_detection import *
# File paths:
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
from sklearn import *
from emoji import UNICODE_EMOJI
import re
from nltk import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Winsorization function
from scipy.stats.mstats import winsorize

# Figures save path
figure_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Report\\Figures\\"

original_datalabels = [
    'user.verified', 
    'user.geo_enabled',
    'user.default_profile', 
    'user.followers_count', 
    'user.friends_count',
    'user.listed_count', 
    'user.favourites_count', 
    'user.statuses_count'
    ]

inferred_datalabels = [
    'user.followers_friends_ratio',
    'user.favourites_statuses_ratio', 
    'user.listed_followers_ratio',
    'user.listed_friends_ratio', 
    'user.followers_statuses_ratio',
    'user.name.length', 
    'user.screen_name.length',
    'user.description.length', 
    'user.name.number_count',
    'user.screen_name.number_count', 
    'user.name.emoji_count',
    'user.description.emoji_count', 
    'user.name.hashtag_count',
    'user.description.hashtag_count', 
    'user.description.url_count',
    'user.description.mention_count',
    #'user.description.vader',
    #'user.name.vader'
    ]

age_datalabels      = [
    'user.age',
    'user.followers_growth_rate', 'user.friends_growth_rate',
    'user.listed_growth_rate', 'user.favourites_growth_rate',
    'user.statuses_growth_rate'
    ]

datalabels = original_datalabels + inferred_datalabels










# 2. RETRIEVING ALL AVAILABLE FEATURES -------------------------------------------

# Functions to count emojis, hashtags, urls and mentions in string variables
def get_emoji_count(x):
    count = 0
    for char in x:
        if char in UNICODE_EMOJI:
            count += 1
    return count

punct = {"!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "}"}

def get_hashtag_count(x):
    count = 0
    for word in x.split():
        if len(word) < 2:
            pass
        elif word[0] == "#" and word[1] not in punct:
            count += 1
    return count    
        
def get_url_count(x):
    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x))

TT = TweetTokenizer(strip_handles=True)

def get_mention_count(x):
    count = 0
    for word in x.split():
        if len(TT.tokenize(word)) == 0:
            count += 1
        else:
            pass
    return count
    
def get_features(age = False, wins = False, vad = False):
    """
    Gets all features of dataframe...
    age = True => include inferred age/growth data as well
    wins = True => winsorize the data (i.e. truncate maximum allowed value to the x-th percentile)
    wins = Fale => maximum allowed value is maximum value that is not infinity
    """
    
    df = pickle.load(open(m4r_data + "balanced_account_training_data.p", "rb"))
    df.loc[:, "user.description"].fillna("", inplace = True)
    # df.loc[75366, "user.name"] = "GoooEmiliooo"
    
    if age:
    # Age related data:
        df["user.age"] = (df["crawled_at"] - df["user.created_at"]).dt.days.replace({0 : 1})
        df["user.followers_growth_rate"] = df["user.followers_count"] / df["user.age"]
        df["user.friends_growth_rate"] = df["user.friends_count"] / df["user.age"]
        df["user.listed_growth_rate"] = df["user.listed_count"] / df["user.age"]
        df["user.favourites_growth_rate"] = df["user.favourites_count"] / df["user.age"]
        df["user.statuses_growth_rate"] = df["user.statuses_count"] / df["user.age"]
    
    
    # Inferred Ratio data:
    if wins:
        L = [0, 0.0001]
        df["user.followers_friends_ratio"] = winsorize( (df["user.followers_count"] / df["user.friends_count"]).fillna(np.inf) , limits = L)
        df["user.favourites_statuses_ratio"] = winsorize( df["user.favourites_count"] / df["user.statuses_count"] )
        df["user.listed_followers_ratio"] = ( df["user.listed_count"] / df["user.followers_count"] )
        df["user.listed_friends_ratio"] = ( df["user.listed_count"] / df["user.friends_count"] )
        df["user.followers_statuses_ratio"] = ( df["user.followers_count"] / df["user.statuses_count"] )
    else:
        df["user.followers_friends_ratio"] = ( df["user.followers_count"] / df["user.friends_count"] ).replace(np.infty, np.nan)
        df["user.followers_friends_ratio"] = df["user.followers_friends_ratio"].fillna(np.max(df["user.followers_friends_ratio"]))
        df["user.favourites_statuses_ratio"] = ( df["user.favourites_count"] / df["user.statuses_count"] ).replace(np.infty, np.nan)
        df["user.favourites_statuses_ratio"] = df["user.favourites_statuses_ratio"].fillna(np.max(df["user.favourites_statuses_ratio"]))
        df["user.listed_followers_ratio"] = ( df["user.listed_count"] / df["user.followers_count"] ).replace(np.infty, np.nan)
        df["user.listed_followers_ratio"] = df["user.listed_followers_ratio"].fillna(np.max(df["user.listed_followers_ratio"]))
        df["user.listed_friends_ratio"] = ( df["user.listed_count"] / df["user.friends_count"] ).replace(np.infty, np.nan)
        df["user.listed_friends_ratio"] = df["user.listed_friends_ratio"].fillna(np.max(df["user.listed_friends_ratio"]))
        df["user.followers_statuses_ratio"] = ( df["user.followers_count"] / df["user.statuses_count"] ).replace(np.infty, np.nan)
        df["user.followers_statuses_ratio"] = df["user.followers_statuses_ratio"].fillna(np.max(df["user.followers_statuses_ratio"]))
    
    
    # Other inferred data: name, screen name, description data:
    df["user.name.length"] = df["user.name"].str.len()
    df["user.name.number_count"] = df["user.name"].str.count(r"\d")
    df["user.name.emoji_count"] = df["user.name"].apply(get_emoji_count)
    df["user.name.hashtag_count"] = df["user.name"].apply(get_hashtag_count)
    
    
    df["user.screen_name.length"] = df["user.screen_name"].str.len()
    df["user.screen_name.number_count"] = df["user.screen_name"].str.count(r"\d")
    
    df["user.description.length"] = df["user.description"].str.len()
    df["user.description.emoji_count"] = df["user.description"].apply(get_emoji_count)
    df["user.description.hashtag_count"] = df["user.description"].apply(get_hashtag_count)
    df["user.description.url_count"] = df["user.description"].apply(get_url_count)
    df["user.description.mention_count"] = df["user.description"].apply(get_mention_count)
    
    
    if vad:
        sid = SentimentIntensityAnalyzer()
        df["user.name.vader"] = df["user.name"].apply(lambda x : sid.polarity_scores(x)["compound"])
        df["user.description.vader"] = df["user.description"].apply(lambda x : sid.polarity_scores(x)["compound"])
    
    
    df.fillna(0, inplace = True)
    
    
    # pickle.dump(df, open(m4r_data + "account_training_data.p", "wb"))
    
    return df












# 3. RFC FEATURE SELECTION ----------------------------------------------------
def rfc_feature_ranking(save = False):
    df = get_features()
    

    scaling = StandardScaler()
    X = scaling.fit_transform(df[datalabels])
    y = df["class"].replace({"human" : 0, "bot" : 1})
    feature_names = df[datalabels].columns

    clf = ensemble.RandomForestClassifier(random_state = 25*17)
    clf.fit(X, y)
    
    importances = clf.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    
    n = len(datalabels)
    plt.figure()
    plt.title("Feature importances from Random Forest Classifier", fontweight = "bold")
    plt.bar(range(n), importances[indices[:n]], color="r", align="center")
    plt.xticks(range(n), feature_names[indices[:n]], rotation = 90)
    plt.xlabel("Feature", fontweight = "bold")
    plt.ylabel("Relative Feature Importance", fontweight = "bold")
    if save:
        plt.savefig(figure_path + "rfc_feature_importance_diagram.pdf", bbox_inches = "tight")
    plt.show()

    
    
# 4. SVD "EXPLAINED VARIANCE RATIO" -------------------------------------------
def svd_feature_ranking(save = False):
    df = get_features()
    
    scaling = StandardScaler()
    X = scaling.fit_transform(df[datalabels])
    y = df["class"].replace({"human" : 0, "bot" : 1})
    feature_names = df[datalabels].columns
    
    n = 22
    svd = decomposition.TruncatedSVD(n_components=n, random_state = 1349*565)
    svd.fit(X)
    
    
    sorted_importance_datalabels = [datalabels[i] for i in svd.explained_variance_ratio_.argsort()[::-1]]
    sorted_importance = np.sort(np.abs(svd.explained_variance_ratio_))[::-1]
    
    plt.bar(range(len(sorted_importance)), sorted_importance)
    plt.xticks(range(len(sorted_importance)), sorted_importance_datalabels, rotation = 90)
    plt.xlabel("Feature", fontweight = "bold")
    plt.ylabel("Explained Variance Ratio", fontweight = "bold") # or "Explained Variance"
    plt.title("Explained Variance from Singular Value Decomposition", fontweight = "bold")
    if save:
        plt.savefig(figure_path + "svd_feature_ranking_diagram.pdf", bbox_inches = "tight")
    plt.show()
    
    
    # OR...
    
    
    
    sorted_importance_datalabels = [datalabels[i] for i in svd.components_[0].argsort()[::-1]]
    sorted_importance = np.sort(np.abs(svd.components_[0]))[::-1]
    
    
    plt.bar(range(len(datalabels)), sorted_importance)
    plt.xticks(range(len(datalabels)), sorted_importance_datalabels, rotation = 90)
    plt.xlabel("Feature", fontweight = "bold")
    plt.ylabel("Feature Score", fontweight = "bold") # or "Explained Variance"
    plt.title("Explained Variance from Singular Value Decomposition", fontweight = "bold")
    if save:
        plt.savefig(figure_path + "svd_feature_ranking_diagram.pdf", bbox_inches = "tight")
    plt.show()
    

# 5. FEATURE SELECTION FROM BUILT IN SKLEARN MODULES --------------------------
def rfe_feature_selection():
    """
    recursive feature elimination
    """
    
    df = get_features()
    

    scaling = StandardScaler()
    X = scaling.fit_transform(df[datalabels])
    y = df["class"].replace({"human" : 0, "bot" : 1})
    feature_names = df[datalabels].columns

    clf = ensemble.RandomForestClassifier(random_state = 25*17*19)
    rfe = feature_selection.RFE(clf, n_features_to_select = 1)
    rfe.fit(X, y)
    
    pd.DataFrame({"feature" : datalabels, "ranking" : rfe.ranking_}).sort_values("ranking")
    
    # [datalabels[x] for x in np.argsort(rfe.ranking_)]
    
    
    
    
    
    

# # Reducing the 'features' and checking...
# def reduced_feature_selection():
#     df = get_all_features()
    
#     scaling = StandardScaler()
#     X = scaling.fit_transform(df[non_age_datalabels])
#     y = df["class"].replace({"human" : 0, "bot" : 1})
#     feature_names = df[non_age_datalabels].columns
    
#     # =========================================================
    
#     clf = ensemble.RandomForestClassifier(random_state = 25*17)
#     clf.fit(X, y)

#     importances = clf.feature_importances_
    
#     indices = np.argsort(importances)[::-1]
    
#     n = len(non_age_datalabels)
#     plt.figure()
#     plt.title("Feature importances from Random Forest Classifier\nTop " + str(n) + " Most Important Features")
#     plt.bar(range(n), importances[indices[:n]], color="r", align="center")
#     plt.xticks(range(n), feature_names[indices[:n]], rotation = 90)
#     plt.xlabel("Feature")
#     plt.ylabel("Feature Importance")
    
    
#     # plt.savefig(figure_path + "rfc_feature_importance_diagram.pdf", bbox_inches = "tight")
    
#     plt.show()
    
    
#     # =========================================================
    
#     num_comps = 10
#     svd = decomposition.TruncatedSVD(n_components = num_comps, random_state = 1349*565)
#     svd.fit(X)
    
#     sorted_importance_datalabels = [non_age_datalabels[i] for i in svd.components_[0].argsort()[::-1]]
#     sorted_importance = np.sort(np.abs(svd.components_[0]))[::-1]
    
#     plt.bar(range(n), sorted_importance[:n])
#     plt.xticks(range(n), sorted_importance_datalabels[:n], rotation = 90)
#     plt.xlabel("Feature")
#     plt.ylabel("Components Importance")
#     plt.title("Feature importances from Singular Value Decomposition\nTop " + str(n) + " Most Important Features")
#     plt.show()
    
def plot_features():
    m = max(NDF["user.friends_count"])
    sns.boxplot(data = NDF, y = "user.friends_count", x = "class");
    plt.yscale("symlog"); plt.ylim([0, m]); plt.show()
    
    m = max(NDF["user.followers_count"])
    sns.boxplot(data = NDF, y = "user.followers_count", x = "class");
    plt.yscale("symlog"); plt.ylim([0, m]); plt.show()
    
    m = max(NDF["user.statuses_count"])
    sns.boxplot(data = NDF, y = "user.statuses_count", x = "class");
    plt.yscale("symlog"); plt.ylim([0, m]); plt.show()

# # 2. PCA ----------------------------------------------------------------------
# X_trn, y_trn = get_data(CRESCI + MIXED, tr = 0, infer = True)
# scaling = StandardScaler()
# trn_PCA = decomposition.PCA(n_components=.95) # find 95% of information...
# trn_features_PCA = trn_PCA.fit_transform(scaling.fit_transform(X_trn))

# #tst_descriptors_PCA = train_PCA.transform(scaling.transform(test_descriptors_full))
# trn_features_PCA.shape



# # 3. 
# X_trn, y_trn = get_data(CRESCI + MIXED, tr = 0, infer = True)
# scaling = StandardScaler()
# all_features = X_trn.columns
# X_trn = scaling.fit_transform(X_trn)

# clf = ensemble.RandomForestClassifier(random_state = 25*17)
# clf.fit(X_trn, y_trn)

# importances = clf.feature_importances_

# std = np.std([tree.feature_importances_ for tree in clf.estimators_],
#          axis=0)

# indices = np.argsort(importances)[::-1]

# plt.figure()
# plt.title("Feature importances from Random Forest Classifier")
# plt.bar(range(len(importances)), importances[indices], color="r", align="center")
# #plt.xticks(range(50), indices, rotation = 90)
# plt.xticks(range(len(indices)), all_features[indices], rotation = 90)
# #plt.xticklabels(indices)
# #plt.xlim([-1, 50])
# plt.show()


# userdict = pickle.load(open(m4r_data + "us_and_georgia_users.p", "rb"))
