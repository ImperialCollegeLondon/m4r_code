"""
Title: Feature Selection for Account Level Detection

Description: Analysis to help decide on which features to use for the account
level detection task. We use three different methods:
    i.   Principal Components Analysis
    ii.  Random Forest Classifier Feature Importance Scores
    iii. Recursive Feature Elimination with Random Forest Classifier
"""


# Path to find training data:
# CHANGE THIS TO THE FOLDER LOCATION OF m4r_data
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"


# 1. SETUP --------------------------------------------------------------------
import pickle, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font="Arial") # plot style
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2, f_classif
from sklearn.preprocessing import StandardScaler, minmax_scale
from emoji import UNICODE_EMOJI
from nltk import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
figure_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Report\\Figures\\" # Figures save path
# Names of the available user features:
datalabels = [
    'user.verified', 
    'user.geo_enabled',
    'user.default_profile', 
    'user.followers_count', 
    'user.friends_count',
    'user.listed_count', 
    'user.favourites_count', 
    'user.statuses_count',
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
    ]
# Non-exhaustive set of punction symbols:
punct = {"!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~", "}"}


# 2. RETRIEVING ALL AVAILABLE FEATURES -------------------------------------------

# Functions to count emojis, hashtags, urls and mentions in a string
def get_emoji_count(x):
    """
    Input: x (string)
    Output: number of emojis in x (int)
    """
    count = 0
    for char in x:
        if char in UNICODE_EMOJI:
            count += 1
    return count

def get_hashtag_count(x):
    """
    Input: x (str)
    Output: number of Twitter hashtags in x (int)
    """
    count = 0
    for word in x.split():
        if len(word) < 2:
            pass
        elif word[0] == "#" and word[1] not in punct:
            count += 1
    return count    
        
def get_url_count(x):
    """
    Input: x (str)
    Output: number of urls in x (int)
    """
    return len(re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x)) # regular expresion (regex) search for a URL

# Tweet Tokeniser object (removes user mentions)
TT = TweetTokenizer(strip_handles=True)
def get_mention_count(x):
    """
    Input: x (str)
    Output: number of user mentions in x (int)
    """
    count = 0
    for word in x.split(): #iterate over each of the words in x
        if len(TT.tokenize(word)) == 0: #if the word is a mention, then the Tweet Tokeniser will remove it
            count += 1
        else:
            pass
    return count
    
def get_features():
    """
    Output: training dataset with additional inferred features (Pandas DataFrame)
    
    Loads the training data
    Retrieves inferred features (e.g. counting the number of emojis in the user description etc)
    Appends this to the training data    
    """
    
    df = pickle.load(open(m4r_data + "balanced_account_training_data.p", "rb")) # load data
    df.loc[:, "user.description"].fillna("", inplace = True) # replaces non-existent descriptions with an empty string
    
    # Inferred Ratio data:
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

    # Other inferred data:
    df["user.name.length"] = df["user.name"].str.len() # length of user name
    df["user.name.number_count"] = df["user.name"].str.count(r"\d") # number of digits in user name
    df["user.name.emoji_count"] = df["user.name"].apply(get_emoji_count) # number of emojis in user name
    df["user.name.hashtag_count"] = df["user.name"].apply(get_hashtag_count) # number of hashtags in user name
    
    df["user.screen_name.length"] = df["user.screen_name"].str.len() # length of user screenname
    df["user.screen_name.number_count"] = df["user.screen_name"].str.count(r"\d") # number of digits in user screenname
    
    df["user.description.length"] = df["user.description"].str.len() # length of user description
    df["user.description.emoji_count"] = df["user.description"].apply(get_emoji_count) # number of emojis in user description
    df["user.description.hashtag_count"] = df["user.description"].apply(get_hashtag_count) # number of hashtags in user description
    df["user.description.url_count"] = df["user.description"].apply(get_url_count) # number of urls in user description
    df["user.description.mention_count"] = df["user.description"].apply(get_mention_count) # number of mentions in user description
    
    df.fillna(0, inplace = True) # replace any NaN values with a 0
    
    return df

# 3. FEATURE SELECTION TECHNIQUES ---------------------------------------------
# # 3.i. SVD "EXPLAINED VARIANCE RATIO"
# def svd_feature_ranking():
#     """
#     Performs Singular Value Decomposition to obtain the scores for how much each
#     feature explains variance in the data (intuitively, the features that contain
#     the most information about the dataset)
#     """
#     df = get_features() # Get training dataset with additional inferred features
#     # Scaling the dataset
#     scaling = StandardScaler()
#     X = scaling.fit_transform(df[datalabels])
#     # Performing Singular Value Decomposition (with n components)
#     n = 22
#     svd = TruncatedSVD(n_components=n, random_state = 1349*565)
#     svd.fit(X)
#     # Retrieving the explained variance ratios
#     sorted_importance_datalabels = [datalabels[i] for i in svd.components_[0].argsort()[::-1]]
#     sorted_importance = np.sort(np.abs(svd.components_[0]))[::-1]
#     # Plotting:    
#     plt.bar(range(len(datalabels)), sorted_importance)
#     plt.xticks(range(len(datalabels)), sorted_importance_datalabels, rotation = 90)
#     plt.xlabel("Feature", fontweight = "bold")
#     plt.ylabel("Explained Variance", fontweight = "bold") # or "Explained Variance"
#     plt.title("Explained Variance from Singular Value Decomposition", fontweight = "bold")
    
def anova_features():
    df = get_features() # Get training dataset with additional inferred features
    # Scaling the dataset
    scaling = StandardScaler()
    X = scaling.fit_transform(df[datalabels])
    y = df["class"].replace({"human" : 0, "bot" : 1})
    feature_names = df[datalabels].columns
    # Performing ANOVA:
    _, p = f_classif(X, y)
    
    X = minmax_scale(df[datalabels], (0,1))
    _, p = chi2(X, y)
    plt.bar(range(len(p)), p)
    

# 3.ii. RFC FEATURE IMPORTANCES
def rfc_feature_importances():
    """
    Trains an RFC model on the entire dataset using all of the available
    features, then retrieving the feature importance scores - i.e. which features
    were the most influential in making decisions
    """
    df = get_features() # Get training dataset with additional inferred features
    # Scaling the dataset
    scaling = StandardScaler()
    X = scaling.fit_transform(df[datalabels])
    y = df["class"].replace({"human" : 0, "bot" : 1})
    feature_names = df[datalabels].columns
    # Creating and fitting an RFC:
    clf = RandomForestClassifier(random_state = 25*17)
    clf.fit(X, y)
    # Retrieving feature importances:
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Plotting:
    n = len(datalabels)
    plt.figure(figsize=(7, 3.2))
    plt.title("Feature importances from Random Forest Classifier", fontweight = "bold")
    plt.bar(range(n), importances[indices[:n]], color="r", align="center")
    plt.xticks(range(n), feature_names[indices[:n]], rotation = 30, ha="right", va="center", rotation_mode = "anchor", fontsize = 7, fontweight = "bold")
    plt.xlabel("Feature", fontweight = "bold")
    plt.ylabel("Relative Feature Importance", fontweight = "bold")
    #plt.savefig("C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Report\\Figures\\rfc_feature_importances.pdf", bbox_inches = "tight")
    plt.show()

# 3.iii. RECURSIVE FEATURE ELIMINATION WITH AN RFC
def rfe_feature_selection():
    """
    Performs recursive feature elimination: trains an RFC multiple times,
    removing the feature with lowest feature importance score after each time.
    Leaves us with a ranking of the features.
    
    WARNING: This may take a while.
    """
    df = get_features() # Get training dataset with additional inferred features
    # Scaling the dataset
    scaling = StandardScaler()
    X = scaling.fit_transform(df[datalabels])
    y = df["class"].replace({"human" : 0, "bot" : 1})
    # Performing Recursive Feature Elimination
    clf = RandomForestClassifier(random_state = 25*17*19)
    rfe = RFE(clf, n_features_to_select = 1)
    rfe.fit(X, y)
    # Printing the ranking of the features:
    print(pd.DataFrame({"feature" : datalabels, "ranking" : rfe.ranking_}).sort_values("ranking"))
    
    