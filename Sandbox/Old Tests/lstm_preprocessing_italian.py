"""
Preprocessing Italian dataset
"""

import sys
import numpy as np
import pickle
from datetime import datetime
from numpy import random
import csv


sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Data Harvesting")
from full_text_tokeniser import text_tokeniser

# File Paths
path_to_italy = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Italian Dataset\\datasets_full.csv\\datasets_full.csv\\"
path_to_m4rdata = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"



def preprocess_genuine_accounts_1(n = None):
    """
    Creates arrays of tweets and corresponding metadata and then pickles them
    Columns of the dataset:
    0:   tweet id
    1:   text
    3:   user id
    12:  retweet count
    13:  reply count
    14:  favourite count
    19:  hashtag count
    20:  url count
    21:  mentions count
    22:  created at (time)
    ---------------------------
    
    """
    path_to_genuine_tweets = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Italian Dataset\\datasets_full.csv\\datasets_full.csv\\genuine_accounts.csv\\tweets.csv"
    samples_genuine = []
    metadata_genuine = []
    with open(path_to_genuine_tweets, "r", encoding = "utf-8") as r:
        reader = csv.reader(r, delimiter = ",")
        s = 0
        for i, line in enumerate(reader):
            sys.stdout.write("\rReading line %i, %i lines skipped" % (i+1, s))
            # The tweet text is given by line[1]
            try:
                p = datetime.strptime(line[-4],"%a %b %d %H:%M:%S %z %Y")
                samples_genuine.append(" ".join(text_tokeniser(line[1])))
                metadata_genuine.append([p.year, p.month, p.day] + [p.hour, p.minute, p.second] + line[-13:-10] + line[-7:-4])
            except:
                s += 1
            if i==n:
                break
    print("\nPickling...")
    pickle.dump(samples_genuine, open(path_to_m4rdata + "samples_italian_genuine.p", "wb"))
    pickle.dump(metadata_genuine, open(path_to_m4rdata + "metadata_italian_genuine.p", "wb"))
    print("Done.")
    
def preprocess_social_spambots_1(n = None):
    """
    ---------------------------
    
    """
    path_to_social_spambots_1 = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Italian Dataset\\datasets_full.csv\\datasets_full.csv\\social_spambots_1.csv\\social_spambots_1.csv\\tweets.csv"
    samples_social_spambots_1 = []
    metadata_social_spambots_1 = []
    with open(path_to_social_spambots_1, "r", encoding = "latin-1") as r:
        reader = csv.reader(r, delimiter = ",")
        s = 0
        for i, line in enumerate(reader):
            sys.stdout.write("\rReading line %i, %i lines skipped" % (i+1, s))
            # The tweet text is given by line[1]
            try:
                p = datetime.strptime(line[-4],"%a %b %d %H:%M:%S %z %Y")
                samples_social_spambots_1.append(" ".join(text_tokeniser(line[1])))
                metadata_social_spambots_1.append([p.year, p.month, p.day] + [p.hour, p.minute, p.second] + line[-13:-10] + line[-7:-4])
            except:
                s += 1
            if i==n:
                break
    print("\nPickling...")
    pickle.dump(samples_social_spambots_1, open(path_to_m4rdata + "samples_italian_social_spambots_1.p", "wb"))
    pickle.dump(metadata_social_spambots_1, open(path_to_m4rdata + "metadata_italian_social_spambots_1.p", "wb"))
    print("Done.")
    
def preprocess_social_spambots_2(n = None):
    """
    ---------------------------
    
    """
    path_to_social_spambots_2 = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Italian Dataset\\datasets_full.csv\\datasets_full.csv\\social_spambots_2.csv\\social_spambots_2.csv\\tweets.csv"
    samples_social_spambots_2 = []
    metadata_social_spambots_2 = []
    with open(path_to_social_spambots_2, "r", encoding = "latin-1") as r:
        reader = csv.reader(r, delimiter = ",")
        s = 0
        for i, line in enumerate(reader):
            sys.stdout.write("\rReading line %i, %i lines skipped" % (i+1, s))
            # The tweet text is given by line[1]
            try:
                p = datetime.strptime(line[-4],"%a %b %d %H:%M:%S %z %Y")
                samples_social_spambots_2.append(" ".join(text_tokeniser(line[1])))
                metadata_social_spambots_2.append([p.year, p.month, p.day] + [p.hour, p.minute, p.second] + line[-13:-10] + line[-7:-4])
            except:
                s += 1
            if i==n:
                break
    print("\nPickling...")
    pickle.dump(samples_social_spambots_2, open(path_to_m4rdata + "samples_italian_social_spambots_2.p", "wb"))
    pickle.dump(metadata_social_spambots_2, open(path_to_m4rdata + "metadata_italian_social_spambots_2.p", "wb"))
    print("Done.")
    
    
def preprocess_social_spambots_3(n = None):
    """
    ---------------------------
    
    """
    path_to_social_spambots_3 = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Italian Dataset\\datasets_full.csv\\datasets_full.csv\\social_spambots_3.csv\\tweets.csv"
    samples_social_spambots_3 = []
    metadata_social_spambots_3 = []
    with open(path_to_social_spambots_3, "r", encoding = "latin-1") as r:
        reader = csv.reader(r, delimiter = ",")
        s = 0
        for i, line in enumerate(reader):
            sys.stdout.write("\rReading line %i, %i lines skipped" % (i+1, s))
            # The tweet text is given by line[1]
            try:
                p = datetime.strptime(line[-4],"%a %b %d %H:%M:%S %z %Y")
                samples_social_spambots_3.append(" ".join(text_tokeniser(line[1])))
                metadata_social_spambots_3.append([p.year, p.month, p.day] + [p.hour, p.minute, p.second] + line[-13:-10] + line[-7:-4])
            except:
                s += 1
            if i==n:
                break
    print("\nPickling...")
    pickle.dump(samples_social_spambots_3, open(path_to_m4rdata + "samples_italian_social_spambots_3.p", "wb"))
    pickle.dump(metadata_social_spambots_3, open(path_to_m4rdata + "metadata_italian_social_spambots_3.p", "wb"))
    print("Done.")
    
def preprocess_traditional_spambots_1(n = None):
    """
    ---------------------------
    
    """
    path_to_traditional_spambots_1 = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Italian Dataset\\datasets_full.csv\\datasets_full.csv\\traditional_spambots_1.csv\\tweets.csv"
    samples_traditional_spambots_1 = []
    metadata_traditional_spambots_1 = []
    with open(path_to_traditional_spambots_1, "r", encoding = "latin-1") as r:
        reader = csv.reader(r, delimiter = ",")
        s = 0
        for i, line in enumerate(reader):
            if i == 0:
                # REMOVING THE HEADER BECAUSE THE TRY: DATE HAS BEEN REMOVED
                if i == n:
                    break
                pass
            else:
                sys.stdout.write("\rReading line %i, %i lines skipped" % (i+1, s))
                # The tweet text is given by line[1]
                try:
                    samples_traditional_spambots_1.append(" ".join(text_tokeniser(line[1])))
                    metadata_traditional_spambots_1.append(line[-13:-10] + line[-7:-4])
                except:
                    s += 1
                if i==n:
                    break
    print("\nPickling...")
    pickle.dump(samples_traditional_spambots_1, open(path_to_m4rdata + "samples_italian_traditional_spambots_1.p", "wb"))
    pickle.dump(metadata_traditional_spambots_1, open(path_to_m4rdata + "metadata_italian_traditional_spambots_1.p", "wb"))
    print("Done.")
    


def creating_shuffled_samples(n = 3000000, validation_split = 0.2, seed = 2771):
    no_of_each = n//2
    no_of_each_bot = no_of_each//4
    samples_human_1 = pickle.load(open(path_to_m4rdata + "samples_italian_genuine.p", "rb"))
    samples_bot_1 = pickle.load(open(path_to_m4rdata + "samples_italian_social_spambots_1.p", "rb"))
    samples_bot_2 = pickle.load(open(path_to_m4rdata + "samples_italian_social_spambots_2.p", "rb"))
    samples_bot_3 = pickle.load(open(path_to_m4rdata + "samples_italian_social_spambots_3.p", "rb"))
    samples_bot_4 = pickle.load(open(path_to_m4rdata + "samples_italian_traditional_spambots_1.p", "rb"))
    metadata_human_1 = pickle.load(open(path_to_m4rdata + "metadata_italian_genuine.p", "rb"))
    metadata_bot_1 = pickle.load(open(path_to_m4rdata + "metadata_italian_social_spambots_1.p", "rb"))
    metadata_bot_2 = pickle.load(open(path_to_m4rdata + "metadata_italian_social_spambots_2.p", "rb"))
    metadata_bot_3 = pickle.load(open(path_to_m4rdata + "metadata_italian_social_spambots_3.p", "rb"))
    metadata_bot_4 = pickle.load(open(path_to_m4rdata + "metadata_italian_traditional_spambots_1.p", "rb"))
    labels = [0]*no_of_each + [1]*no_of_each
    
    
    # First Shuffle
    a,b,c,d,e = [13*seed, 7*seed, 3*seed, 29*seed, 31*seed]
    
    rng = np.random.RandomState(a)
    rng.shuffle(samples_human_1)
    rng = np.random.RandomState(a)
    rng.shuffle(metadata_human_1)
    
    rng = np.random.RandomState(b)
    rng.shuffle(samples_bot_1)
    rng = np.random.RandomState(b)
    rng.shuffle(metadata_bot_1)
    
    rng = np.random.RandomState(c)
    rng.shuffle(samples_bot_2)
    rng = np.random.RandomState(c)
    rng.shuffle(metadata_bot_2)
    
    rng = np.random.RandomState(d)
    rng.shuffle(samples_bot_3)
    rng = np.random.RandomState(d)
    rng.shuffle(metadata_bot_3)
    
    rng = np.random.RandomState(e)
    rng.shuffle(samples_bot_4)
    rng = np.random.RandomState(e)
    rng.shuffle(metadata_bot_4)
    
    # Selecting no_of_each number of samples in a 50:50 split
    samples = []
    metadata = []
    # Selecting human tweet samples
    samples += samples_human_1[:no_of_each]
    metadata += metadata_human_1[:no_of_each]
    if len(samples) != no_of_each:
        print(len(samples))
        raise Exception("The value of n is too high,\n not enough human tweets")
    # Selecting no_of_each_bot number of bot tweet samples from each category:
    samples += samples_bot_1[:no_of_each_bot]
    metadata += metadata_bot_1[:no_of_each_bot]
    print(len(metadata))
    print(len(samples))
    samples += samples_bot_2[:no_of_each_bot]
    metadata += metadata_bot_2[:no_of_each_bot]
    print(len(metadata))
    print(len(samples))
    samples += samples_bot_3[:no_of_each_bot]
    metadata += metadata_bot_3[:no_of_each_bot]
    print(len(metadata))
    print(len(samples))
    samples += samples_bot_4[:no_of_each_bot]
    metadata += metadata_bot_4[:no_of_each_bot]
    print(len(metadata))
    print(len(samples))
    
    if len(samples) != n:
        l = len(samples)
        samples += samples_bot_1[no_of_each_bot:no_of_each_bot+(n - l)]
        metadata += metadata_bot_1[no_of_each_bot:no_of_each_bot+(n - l)]
        if len(samples) != n:
            raise Exception("The value of n is too high, \n not enough bot tweets")
    
    # adding the remaining leftover bot samples:
    
    
    # Second Shuffle
    # *shuffling*
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
    rng = np.random.RandomState(seed)
    rng.shuffle(metadata)
    print("=======================")
    print(len(samples))
    print(len(metadata))

    # *splitting*
    num_validation_samples = int(validation_split * len(samples))
    train_samples = samples[:-num_validation_samples]
    val_samples = samples[-num_validation_samples:]
    train_labels = labels[:-num_validation_samples]
    val_labels = labels[-num_validation_samples:]
    train_metadata = metadata[:-num_validation_samples]
    val_metadata = metadata[-num_validation_samples:]
    
    pickle.dump(train_samples, open(path_to_m4rdata+"italian_train_samples.p", "wb" ))
    pickle.dump(val_samples, open(path_to_m4rdata+"italian_val_samples.p", "wb" ))
    pickle.dump(train_labels, open(path_to_m4rdata+"italian_train_labels.p", "wb" ))
    pickle.dump(val_labels, open(path_to_m4rdata+"italian_val_labels.p", "wb" ))
    pickle.dump(train_metadata, open(path_to_m4rdata+"italian_train_metadata.p", "wb" ))
    pickle.dump(val_metadata, open(path_to_m4rdata+"italian_val_metadata.p", "wb" ))
    

    return train_samples, val_samples, train_labels, val_labels, train_metadata, val_metadata
        

    
    
    
    