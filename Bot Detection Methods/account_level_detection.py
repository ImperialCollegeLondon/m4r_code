"""
Account Level detection

Contents:
1. Setup
2. Misc.
3. "Get" Functions
4. Comparing Classifiers
5. Plotting
"""
# 1. SETUP --------------------------------------------------------------------
import pickle #, math, sys, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Packages for synthetic minority oversampling
from imblearn.combine import SMOTEENN, SMOTETomek
# Packages for models/classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.random import set_seed
# Setting the plot style
sns.set(font="Arial")

# 2. MISC ---------------------------------------------------------------------
# Path to find training data:
m4r_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
# List of dataset names:
DATASETS = [
    'Astroturf 2020',
    'Botometer Feedback 2019',
    'Botwiki 2019',
    'Celebrity 2019',
    'Cresci 2017 Genuine Users',
    'Cresci 2017 Social Spambots 1',
    'Cresci 2017 Social Spambots 2',
    'Cresci 2017 Social Spambots 3',
    'Cresci 2017 Traditional Spambots 1',
    'Cresci RTbust 2019',
    'Cresci Stock 2018',
    'Gilani 2017',
    'Midterm 2018',
    'Political Bots 2019',
    'Varol 2017',
    'Verified 2019'
    ]
# Dataset mixtures as outlined in report
D1 = [x for x in DATASETS if x[:7] != "Midterm"]
D2 = ['Cresci 2017 Genuine Users', 'Cresci 2017 Social Spambots 1', 'Cresci 2017 Social Spambots 2', 'Cresci 2017 Social Spambots 3']
D3 = ["Midterm 2018"]
# List of 12 features we will use for account level detection:
features = [
    'user.followers_count', 
    'user.friends_count',
    'user.listed_count', 
    'user.favourites_count', 
    'user.statuses_count',
    'user.verified',
    'user.geo_enabled',
    'user.default_profile',
    "user.favourites_statuses_ratio",
    "user.followers_statuses_ratio",
    "user.followers_friends_ratio",
    "user.description.length"
    ]

# 3. FUNCTIONS ----------------------------------------------------------------
def get_full_dataset(df = None, return_maxes = False):
    # Loading the balanced user training data:
    if df is None:
        df = pickle.load(open(m4r_data + "balanced_account_training_data.p", "rb"))
    # Inferring the ratio features...
    # #followers / #friends
    df["user.followers_friends_ratio"] = ( df["user.followers_count"] / df["user.friends_count"] ).replace(np.infty, np.nan)
    max_followers_friends_ratio = np.max(df["user.followers_friends_ratio"])
    df["user.followers_friends_ratio"] = df["user.followers_friends_ratio"].fillna(np.max(df["user.followers_friends_ratio"]))
    # #favourites / #statuses
    df["user.favourites_statuses_ratio"] = ( df["user.favourites_count"] / df["user.statuses_count"] ).replace(np.infty, np.nan)
    max_favourites_statuses_ratio = np.max(df["user.favourites_statuses_ratio"])
    df["user.favourites_statuses_ratio"] = df["user.favourites_statuses_ratio"].fillna(np.max(df["user.favourites_statuses_ratio"]))
    # #followers / #statuses
    df["user.followers_statuses_ratio"] = ( df["user.followers_count"] / df["user.statuses_count"] ).replace(np.infty, np.nan)
    max_followers_statuses_ratio = np.max(df["user.followers_statuses_ratio"])
    df["user.followers_statuses_ratio"] = df["user.followers_statuses_ratio"].fillna(np.max(df["user.followers_statuses_ratio"]))
    # Inferring description length
    df.loc[:, "user.description"].fillna("", inplace = True)
    df["user.description.length"] = df["user.description"].str.len()
    # Filling in nan values
    df = df.fillna(0)
    if return_maxes:
        return df, max_followers_friends_ratio, max_favourites_statuses_ratio, max_followers_statuses_ratio
    else:
        return df

def get_X_y(mix , df = None , tr = 0.2 , seed = int(1349 * 565 // 13)):
    if df is None:
        df = get_full_dataset()
    D = df[df["dataset"].isin(mix)].sample(frac = 1.0, random_state = seed).reset_index(drop = True)
    if tr == 0:
        return D[features], D["class"].replace({"bot" : 1, "human" : 0})
    else:
        split = int( (1.0 - tr ) * len(D))
        X_trn = D.loc[D.index[:split], features]
        X_tst = D.loc[D.index[split:], features]
        y_trn = D.loc[D.index[:split],"class"].replace({"bot" : 1, "human" : 0}) 
        y_tst = D.loc[D.index[split:],"class"].replace({"bot" : 1, "human" : 0}) 
        return X_trn, y_trn, X_tst, y_tst

def get_smotenn(X_trn, y_trn, seed = int(623*449)):
    SME = SMOTEENN(random_state = seed)
    X_trn, y_trn = SME.fit_resample(X_trn, y_trn)
    return X_trn, y_trn

def get_smotetomek(X_trn, y_trn, seed = int(623*4413)):
    SMTMK = SMOTETomek(random_state = seed)
    X_trn, y_trn = SMTMK.fit_resample(X_trn, y_trn)
    return X_trn, y_trn

def get_scores(clf, X_trn, y_trn, X_tst, y_tst, X_out = None, y_out = None):
    """
    Retrieves the scores of the classifier clf
    """
    # Making predictions...
    p_trn = np.round(clf.predict(X_trn))
    p_tst = np.round(clf.predict(X_tst))
    # Empty list to store scores
    scores = []
    # Appending training score:
    scores += [accuracy_score(y_trn, p_trn)]
    # Appending testing scores:
    scores += [accuracy_score(y_tst, p_tst)]
    scores += [precision_score(y_tst, p_tst)]
    scores += [recall_score(y_tst, p_tst)]
    scores += [f1_score(y_tst, p_tst)]
    # Appending OOS scores (if there is an OOS)
    if X_out is not None:
        p_out = np.round(clf.predict(X_out))
        # Appending out of sample scores:
        scores += [accuracy_score(y_out, p_out)]
        scores += [precision_score(y_out, p_out)]
        scores += [recall_score(y_out, p_out)]
        scores += [f1_score(y_out, p_out)]
    return scores

def get_nn(nfeats):
    """
    Creates the neural network model
    nfeats = X_trn.shape[1] (i.e. number of features to look at)
    """
    clf = Sequential()
    clf.add(Dense(2 * nfeats, input_dim = nfeats, activation = "relu"))
    clf.add(Dropout(0.2))
    clf.add(Dense(4 * nfeats, activation = "relu"))
    clf.add(Dropout(0.2))
    clf.add(Dense(nfeats, activation = "relu"))
    clf.add(Dropout(0.2))
    clf.add(Dense(2, activation = "relu"))
    clf.add(Dropout(0.2))
    clf.add(Dense(1, activation = "sigmoid"))
    clf.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    return clf

# 4. COMPARING CLASSIFIERS ----------------------------------------------------

def compare_italy_to_d1(save = False):
    """
    Comparing training on JUST the Cresci Italian dataset to the D1 dataset
    And compare OUT OF SAMPLE accuracy.
    Explain that although the Cresci Italian dataset may have higher training
    and testing scores, it is symptomatic of overfitting...
    
    ONLY USING ADABOOST...
    """
    scaling = StandardScaler()
    
    # Defining Classifiers
    clf  = AdaBoostClassifier(n_estimators = 50, random_state = 25)
    
    # Names for pandas dataframe
    criterion_names_ = ["Accuracy", "Precision", "Recall", "F1"]
    criterion_names  = ["Train accuracy"] + ["Test " + i for i in criterion_names_] + ["OOS " + i for i in criterion_names_]
    
    # Stratified K Fold
    df = get_full_dataset()
    X1, y1 = get_X_y(D1, df, tr = 0, seed = 1097)
    X2, y2 = get_X_y(D2, df, tr = 0, seed = 1097)
    X_OOS, y_oos = get_X_y(D3, df, tr = 0, seed = 2077)
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1349*5*65*17)
    
    score_dataframe = pd.DataFrame()
    
    # Repeating training & testing on the different splits...
    # D1
    f = 0
    for trn_index, tst_index in skf.split(X1, y1):
        f += 1
        # Stratified Split
        X_trn, X_tst = X1.iloc[trn_index], X1.iloc[tst_index]
        y_trn, y_tst = y1.iloc[trn_index], y1.iloc[tst_index]
        X_oos = X_OOS.copy() # so we don't "retransform" X_oos each time
        
        X_trn = scaling.fit_transform(X_trn)
        X_tst = scaling.transform(X_tst)
        X_oos = scaling.transform(X_oos)
        
        sf = pd.DataFrame(columns = ["Training Set", "Fold" , "Criterion" ,"Score"])
        sf["Training Set"] = ["D1"] * 9
        sf["Fold"] = [f] * 9
        sf["Criterion"] = criterion_names
        
        clf.fit(X_trn, y_trn)
        sf["Score"] = get_scores(clf, X_trn, y_trn, X_tst, y_tst, X_oos, y_oos)
        
        score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
    
    # CRESCI17
    f = 0
    for trn_index, tst_index in skf.split(X2, y2):
        f += 1
        # Stratified Split
        X_trn, X_tst = X2.iloc[trn_index], X2.iloc[tst_index]
        y_trn, y_tst = y2.iloc[trn_index], y2.iloc[tst_index]
        X_oos = X_OOS.copy() # so we don't "retransform" X_oos each time
        
        X_trn = scaling.fit_transform(X_trn)
        X_tst = scaling.transform(X_tst)
        X_oos = scaling.transform(X_oos)
        
        sf = pd.DataFrame(columns = ["Training Set", "Fold" , "Criterion" ,"Score"])
        sf["Training Set"] = ["D2"] * 9
        sf["Fold"] = [f] * 9
        sf["Criterion"] = criterion_names
        
        clf.fit(X_trn, y_trn)
        sf["Score"] = get_scores(clf, X_trn, y_trn, X_tst, y_tst, X_oos, y_oos)
        
        score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
    
    
    
    
    Title = "Comparison of Account Level Detection Models Trained On Different Datasets with 5 Fold Cross Validation\n"
    Title += "(Using AdaBoost Classifier; OOS = Balanced Midterm 2018)"
    ax = sns.barplot(
        data = score_dataframe,
        x="Criterion",
        y="Score",
        hue = "Training Set"
    )
    ax.set_title(Title, fontweight = "bold");
    ax.set_xlabel("Score Criterion", fontweight = "bold")
    ax.set_ylabel("Score", fontweight = "bold")
    ax.legend(title = "Training\nDataset", loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.xticks(rotation=15)
    plt.show()
    
    return score_dataframe

# Cross validation model comparison
def cross_validation_scores(trnsample = D1, outofsample = D3, save = False):
    scaling = StandardScaler()
    
    # Defining Classifiers
    rfc = RandomForestClassifier(random_state = 25)
    lr  = LogisticRegression(max_iter = 1000, random_state=25, penalty='l2')
    sgd = SGDClassifier(max_iter=1000, tol=1e-8, random_state = 25)
    ab  = AdaBoostClassifier(n_estimators = 50, random_state = 25)
    nn  = get_nn(len(features)) # i.e. nfeats
    classifiers = [rfc, lr, sgd, ab, nn]
    
    # Names for pandas dataframe
    classifier_names = ["RFC", "LR", "SVM", "AB", "NN"]
    criterion_names_ = ["Accuracy", "Precision", "Recall", "F1"]
    criterion_names  = ["Train accuracy"] + ["Test " + i for i in criterion_names_] + ["OOS " + i for i in criterion_names_]
    
    # Stratified K Fold
    df = get_full_dataset()
    X, y = get_X_y(trnsample, df, tr = 0, seed = 1097)
    X_OOS, y_oos = get_X_y(outofsample, df, tr = 0, seed = 2077)
    skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1349*5*65*17)
    
    score_dataframe = pd.DataFrame()
    
    # Repeating training & testing on the different splits...
    f = 0
    for trn_index, tst_index in skf.split(X, y):
        f += 1
        # Stratified Split
        X_trn, X_tst = X.iloc[trn_index], X.iloc[tst_index]
        y_trn, y_tst = y.iloc[trn_index], y.iloc[tst_index]
        X_oos = X_OOS.copy() # so we don't "retransform" X_oos each time
        
        X_trn = scaling.fit_transform(X_trn)
        X_tst = scaling.transform(X_tst)
        X_oos = scaling.transform(X_oos)
        # Iterating over the different classifiers...
        
        for clf, clf_name in zip(classifiers, classifier_names):
            print("Calculating ", f, ": ", clf_name)
            sf = pd.DataFrame(columns = ["Model", "Fold" , "Criterion" ,"Score"])
            sf["Model"] = [clf_name] * 9
            sf["Fold"] = [f] * 9
            sf["Criterion"] = criterion_names
            if clf_name == "NN":
                set_seed(25)
                clf.fit(X_trn, y_trn, epochs = 15, batch_size = 32, verbose = 0)
            else:
                clf.fit(X_trn, y_trn)
            sf["Score"] = get_scores(clf, X_trn, y_trn, X_tst, y_tst, X_oos, y_oos)
            score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
    
    Title = "Comparison of Account Level Detection Models with 5 Fold Cross Validation\n"
    Title += "(trained on all datasets except Midterm 2018 (with 80:20 split), OOS = Balanced Midterm 2018)"
    ax = sns.barplot(
        data = score_dataframe,
        x="Criterion",
        y="Score",
        hue = "Model"
    )
    ax.set_title(Title, fontweight = "bold");
    ax.set_xlabel("Score Criterion", fontweight = "bold")
    ax.set_ylabel("Score", fontweight = "bold")
    ax.legend(title = "Model", loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.xticks(rotation=15)
    plt.show()
    return score_dataframe
    

# Resampling Comparison...
def compare_resampling(save = False):
    clf = AdaBoostClassifier(n_estimators = 50, random_state = int(25*123817 // 7))
    
    scaling = StandardScaler()
    
    sampling         = ["None", "SMOTENN", "SMOTETomek"]
    criterion_names_ = ["Accuracy", "Precision", "Recall", "F1"]
    criterion_names  = ["Train accuracy"] + ["Test " + i for i in criterion_names_]
    
    df = get_full_dataset()
    X_TRN, y_TRN, X_TST, y_tst = get_X_y(DATASETS, df, tr = 0.2, seed = int(1097*9127))
    
    score_dataframe = pd.DataFrame()
    
    for sam in sampling:
        if sam == "SMOTENN":
            SME = SMOTEENN(random_state = int(15571*13 // 3))
            X_trn, y_trn = SME.fit_resample(X_TRN, y_TRN)
        elif sam == "SMOTETomek":
            SMTMK = SMOTETomek(random_state = int((3*14139841 - 3)//3))
            X_trn, y_trn = SMTMK.fit_resample(X_TRN, y_TRN)
        else:
            X_trn = X_TRN.copy()
            y_trn = y_TRN.copy()
        X_trn = scaling.fit_transform(X_trn)
        X_tst = scaling.transform(X_TST)
        
        clf.fit(X_trn, y_trn)
        
        sf = pd.DataFrame(columns = ["Resampling" , "Criterion" ,"Score"])
        sf["Resampling"] = [sam] * 5
        sf["Criterion"] = criterion_names
        sf["Score"] = get_scores(clf, X_trn, y_trn, X_tst, y_tst)
        score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
        
    
    Title = "Comparison of Account Level Detection Resampling Techniques with AdaBoost\n"
    Title += "(trained on all datasets)"
    ax = sns.barplot(
        data = score_dataframe,
        x="Criterion",
        y="Score",
        hue = "Resampling"
    )
    ax.set_title(Title, fontweight = "bold");
    ax.set_xlabel("Score Criterion", fontweight = "bold")
    ax.set_ylabel("Score", fontweight = "bold")
    ax.legend(title = "Resampling\nMethod", loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    plt.xticks(rotation=15)
    plt.show()
        
        
    return score_dataframe
    




# 5. APPLYING THE SUITABLE CLASSIFIER TO OUR COLLECTED DATA -------------------

def applying_classifier_to_all_accounts():
    df = pickle.load(open(m4r_data + "us_and_georgia_accounts.p", "rb"))
    
    trainset = get_full_dataset()
    users = get_full_dataset(df)
    
    X_users = users[features]
    
    X_trn = trainset[features]
    y_trn = trainset["class"].replace({"bot" : 1, "human" : 0})
                                      
    SME = SMOTEENN(random_state = 2727841)
    X_trn, y_trn = SME.fit_resample(X_trn, y_trn)
    
    scaling = StandardScaler()
    X_trn = scaling.fit_transform(X_trn)
    X_users  = scaling.transform(X_users)

    clf = AdaBoostClassifier(n_estimators = 50, random_state = 9926737)
    clf.fit(X_trn, y_trn)    
    
    p_users = np.round(clf.predict(X_users))
    
    users["predicted_class"] = p_users
    
    users["predicted_class"] = users["predicted_class"].replace({0 : "human", 1 : "bot"})
    
    print("% Bots (ALL): ", sum(p_users)/len(p_users))
    
    # adding predicted class column to df
    df = df.merge(users[["user.id", "predicted_class"]], how = "left", on = "user.id")
    
    # adding donald trump to users
    d_row = {"user.id" : 25073877, "user.name" : "realDonaldTrump", "user.screen_name" : "realDonaldTrump", "user.verified" : True, "predicted_class"  : "human"}
    df = df.append(d_row, ignore_index=True)
    
    # saving to csv
    #pickle.dump(df, open(m4r_data + "us_and_georgia_accounts.p", "wb"))
