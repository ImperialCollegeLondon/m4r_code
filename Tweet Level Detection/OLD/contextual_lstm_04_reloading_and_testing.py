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

checkpoint_filepath = path_to_m4r_code_tld + "contextual_lstm_model_weights_training_checkpoints_4_million\\"

#model = contextual_lstm_model_1()

#model.load_weights(checkpoint_filepath)

# Preprocessing: # n = 2500000 may be too much!
df_entire = s00_load_data()

df_trn, df_val = s01_split_data(df_entire, n = 2000000, split = 0.8)
vectorizer = s02_make_vectorizer(df_trn, output_sequence_length = 50) # don't need to rerun these because these take aaaaaages
embeddings_index = s03_embeddings_index()
embedding_matrix = s04_embedding_matrix(vectorizer, embeddings_index)

# Vectorizing testing data in batches
vectorized_testing_data_batches = []
for i in range(int(np.ceil(len(df_val)/128))):
    vectorized_testing_data_batches.append(vectorizer(df_val["text"][128*i:128*(i+1)].tolist()))
val_tweet_vector = tf.concat(vectorized_testing_data_batches, axis = 0)