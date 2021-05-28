# Modules
import sys, pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
sys.path.insert(1, "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection")
from contextual_lstm_01_data_preprocessing import *
from contextual_lstm_02_model_structures import *

# File paths
path_to_m4rdata = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
path_to_data_as_pandas = path_to_m4rdata+"Italian Data as Pandas\\"
path_to_glove_file = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\GloVe\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
path_to_m4rcode = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\"
path_to_m4r_code_tld = path_to_m4rcode + "Tweet Level Detection\\"


# Preprocessing: # n = 2500000 may be too much!
df_entire = s00_load_data()

df_trn, df_val = s01_split_data(df_entire, n = 2000000, split = 0.8)
vectorizer = s02_make_vectorizer(df_trn) # don't need to rerun these because these take aaaaaages
embeddings_index = s03_embeddings_index()
embedding_matrix = s04_embedding_matrix(vectorizer, embeddings_index)

# Vectorizing training data in batches
vectorized_training_data_batches = []
for i in range(int(np.ceil(len(df_trn)/128))):
    vectorized_training_data_batches.append(vectorizer(df_trn["text"][128*i:128*(i+1)].tolist()))
trn_tweet_vector = tf.concat(vectorized_training_data_batches, axis = 0)



# Check point:
# Model weights will be saved if they are the best seen so far (in terms of validation accuracy?)
checkpoint_filepath = path_to_m4r_code_tld + "contextual_lstm_model_weights_training_checkpoints_4_million\\"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    #save_freq = "epoch",
    monitor='final_output_acc',
    mode='max',
    save_best_only=True)





# Loading the model structure:
model = contextual_lstm_model_1()

# loading previous model weights:
# model.load_weights(checkpoint_filepath)

# Training the model on very few epochs
second_split = int(0.8*len(trn_tweet_vector))
train_s = trn_tweet_vector[:second_split,:]
train_m = df_trn.iloc[:second_split,2:8].astype(int).values
train_l = df_trn.iloc[:second_split,-1].values
test_s = trn_tweet_vector[second_split:,:]
test_m = df_trn.iloc[second_split:,2:8].astype(int).values
test_l = df_trn.iloc[second_split:,-1].values
num_epochs = 10
batch_size = 512
history = model.fit(
    {'tweet_input': train_s, 'metadata_input': train_m},
    {'final_output': train_l, 'auxiliary_output': train_l},
    validation_data = (
        {'tweet_input': test_s, 'metadata_input': test_m},
        {'final_output': test_l, 'auxiliary_output': test_l}
        ),
    epochs = num_epochs,
    batch_size = batch_size,
    callbacks=[model_checkpoint_callback]
    )



