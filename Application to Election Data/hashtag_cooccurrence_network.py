"""
Building a hashtag cooccurence network
"""

import pandas as pd
import sys, pickle
import operator
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_repository")
from account_level_detection import get_full_dataset, features


# File Path:
    
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"

def get_hashtags(t):
    indices = [i for i, x in enumerate(t) if x == "<hashtag>"]
    return sorted([t[i+1].lower() for i in indices])

def get_cooccurrence_matrix(df):
    """
    Returns a cooccurrence matrix: a dictionary of dictionaries
    say we have a hashtag pair (#apples, #bananas)
    the outer dictionary will contain apples,
    then the inner dictionary will contain bananas : 1 (the count that they appear together)
    """
    H = {} # initialising an empty matrix
    for h in df: # Iterating through each of the tweets hashtags
        for i in range(len(h)-1): # Triangular iteration since (A, B) = (B, A)
            if h[i] not in H.keys(): # adding an empty entry if it is not currently present in the outer dictionary
                H[h[i]] = {}
            for j in range(i+1, len(h)):
                if h[j] not in H[h[i]].keys():
                    H[h[i]][h[j]] = 1 # initialising count
                else:
                    H[h[i]][h[j]] += 1 # updating count
    return H

def distribution_of_cooccurrences(H, TH = 5):
    distribution_of_cooccurrences = []
    for h in H:
        distribution_of_cooccurrences += list(H[h].values())
    distribution_of_cooccurrences = np.array(distribution_of_cooccurrences)
    print("Number of edges:")
    print(">5;   ", sum(distribution_of_cooccurrences > 5))
    print(">100: ", sum(distribution_of_cooccurrences > 100))
    print(">1000:", sum(distribution_of_cooccurrences > 1000))
    print(">2000:", sum(distribution_of_cooccurrences > 2000))
    print(">TH:  ", sum(distribution_of_cooccurrences > TH))
    
def get_most_occurring(H, TH = 5):
    # Making a dataframe from the pandas dictionary... THESE WILL BE THE EDGES
    Hdf = pd.DataFrame(columns = ["Source", "Target", "Weight"])
    tot = len(H)
    for c, h in enumerate(H):
        sys.stdout.write("\r"+str(c+1)+" out of "+str(tot))
        # More than TH cooccurrences        
        valid_hashtag_indices = np.argwhere(np.array(list(H[h].values())) > TH).flatten()
        
        if len(valid_hashtag_indices) == 0:
            pass
        else:
            Hdf = Hdf.append(pd.DataFrame({"Source" : [h] * len(valid_hashtag_indices), "Target" : [list(H[h].keys())[z] for z in valid_hashtag_indices], "Weight" : [list(H[h].values())[z] for z in valid_hashtag_indices] }))
    return Hdf
    
    
    
def classify_accounts():
    df_us = pickle.load(open(m4r_data + "us_election_tweets.p", "rb"))
    us = df_us.groupby("user.id").first().reset_index()
    df_ga = pickle.load(open(m4r_data + "georgia_election_tweets.p", "rb"))
    ga = df_ga.groupby("user.id").first().reset_index()    
    
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
    
    return df_us, df_ga

  
    
    
def build_3(df_us):
    """
    SPLITTING us INTO BOTS-HUMANS ---AND--- POSITIVE NEusTIVE
    """
    
    # # # # # HUMAN us TWEETS # # # # # 
    # Positive ----------------------------------------------------------------
    # Extracting tweets with positive (> 0.25) AND predicted human and more than 1 hashtag
    us_human_pos = df_us[(df_us["vader"] > 0.25) & ((df_us["predicted_class"] == "human") & (df_us["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    us_human_pos = us_human_pos["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    us_human_pos = us_human_pos.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(us_human_pos)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    us_human_pos_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    us_human_pos_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "us_hashtag_cooccurrence_network_human_and_positive_top_25.csv", index = False)
    # Negative ----------------------------------------------------------------
    # Extracting tweets with negative (< - 0.25) AND predicted human and more than 1 hashtag
    us_human_neg = df_us[(df_us["vader"] < -0.25) & ((df_us["predicted_class"] == "human") & (df_us["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    us_human_neg = us_human_neg["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    us_human_neg = us_human_neg.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(us_human_neg)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    us_human_neg_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    us_human_neg_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "us_hashtag_cooccurrence_network_human_and_negative_top_25.csv", index = False)
    
    
    
    # # # # # BOT us TWEETS # # # # # 
    # Positive ----------------------------------------------------------------
    # Extracting tweets with positive (> 0.25) AND predicted human and more than 1 hashtag
    us_bot_pos = df_us[(df_us["vader"] > 0.25) & ((df_us["predicted_class"] == "bot") & (df_us["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    us_bot_pos = us_bot_pos["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    us_bot_pos = us_bot_pos.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(us_bot_pos)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    us_bot_pos_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    us_bot_pos_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "us_hashtag_cooccurrence_network_bot_and_positive_top_25.csv", index = False)
    # Negative ----------------------------------------------------------------
    # Extracting tweets with negative (< - 0.25) AND predicted bot and more than 1 hashtag
    us_bot_neg = df_us[(df_us["vader"] < -0.25) & ((df_us["predicted_class"] == "bot") & (df_us["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    us_bot_neg = us_bot_neg["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    us_bot_neg = us_bot_neg.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(us_bot_neg)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    us_bot_neg_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    us_bot_neg_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "us_hashtag_cooccurrence_network_bot_and_negative_top_25.csv", index = False)
    
    
    
    
    
    
    
    

def build_4(df_ga):
    """
    SPLITTING GA INTO BOTS-HUMANS ---AND--- POSITIVE NEGATIVE
    """
    
    # # # # # HUMAN GA TWEETS # # # # # 
    # Positive ----------------------------------------------------------------
    # Extracting tweets with positive (> 0.25) AND predicted human and more than 1 hashtag
    ga_human_pos = df_ga[(df_ga["vader"] > 0.25) & ((df_ga["predicted_class"] == "human") & (df_ga["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    ga_human_pos = ga_human_pos["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    ga_human_pos = ga_human_pos.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(ga_human_pos)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    ga_human_pos_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    ga_human_pos_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "ga_hashtag_cooccurrence_network_human_and_positive_top_25.csv", index = False)
    # Negative ----------------------------------------------------------------
    # Extracting tweets with negative (< - 0.25) AND predicted human and more than 1 hashtag
    ga_human_neg = df_ga[(df_ga["vader"] < -0.25) & ((df_ga["predicted_class"] == "human") & (df_ga["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    ga_human_neg = ga_human_neg["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    ga_human_neg = ga_human_neg.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(ga_human_neg)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    ga_human_neg_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    ga_human_neg_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "ga_hashtag_cooccurrence_network_human_and_negative_top_25.csv", index = False)
    
    
    
    # # # # # BOT GA TWEETS # # # # # 
    # Positive ----------------------------------------------------------------
    # Extracting tweets with positive (> 0.25) AND predicted human and more than 1 hashtag
    ga_bot_pos = df_ga[(df_ga["vader"] > 0.25) & ((df_ga["predicted_class"] == "bot") & (df_ga["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    ga_bot_pos = ga_bot_pos["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    ga_bot_pos = ga_bot_pos.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(ga_bot_pos)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    ga_bot_pos_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    ga_bot_pos_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "ga_hashtag_cooccurrence_network_bot_and_positive_top_25.csv", index = False)
    # Negative ----------------------------------------------------------------
    # Extracting tweets with negative (< - 0.25) AND predicted bot and more than 1 hashtag
    ga_bot_neg = df_ga[(df_ga["vader"] < -0.25) & ((df_ga["predicted_class"] == "bot") & (df_ga["hashtag_count"] > 1))]
    # Splitting the text into a list of words/tokens
    ga_bot_neg = ga_bot_neg["tokenised_text"].str.split()
    # Extracting hashtags (i.e. keeping only the token following "<hashtag>")
    ga_bot_neg = ga_bot_neg.apply(lambda x : get_hashtags(x))
    # Building cooccurrence matrix:
    H = get_cooccurrence_matrix(ga_bot_neg)
    # Printing out distribution
    distribution_of_cooccurrences(H, TH = 5)
    # Get dataframe of most common hashtags
    ga_bot_neg_rank = get_most_occurring(H, TH = 5)
    # TOP 25 COOCCURRENCES:
    ga_bot_neg_rank.sort_values("Weight").tail(25).to_csv(m4r_data + "ga_hashtag_cooccurrence_network_bot_and_negative_top_25.csv", index = False)
    
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    