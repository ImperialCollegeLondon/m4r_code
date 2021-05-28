# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:41:44 2021

@author: fangr
"""

import pickle, math, sys, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

collected_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Collected Data\\"

# df = pickle.load(open(collected_data + "us_election_tweets.p", "rb"))

# Default styles...
sns.set(font="Arial")

sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection")
from tweet_level_detection import contextual_lstm_model
from tweet_level_detection_preprocessing_pt_3 import load_sample, tensorfy, create_embeddings_index, create_embedding_matrix

def plot_1(df):
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
    
    