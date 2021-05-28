"""
Tweet Level Detection:
Contextual LSTM but ONLY with the contextual features and not account level features
Contents:
1. Packages
2. File paths
3. Preprocessing 
4. Model Structure
5. Training Model
"""



# 1. Packages -----------------------------------------------------------------
import sys, pickle, time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.initializers import Constant
from tensorflow.keras.backend import cast
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.preprocessing import StandardScaler

# 2. File Paths ---------------------------------------------------------------
indiana = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Indiana Dataset\\"
italy   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Cresci Italian Dataset\\"
other   = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\Other Datasets\\"
lstmdata= "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\LSTM Training Data\\"
glove_file = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
weightsavepath = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection\\contextual_LSTM_weights\\"

# 3. Preprocessing ------------------------------------------------------------
def load_sample(n = 200000):
    """
    5 Million tweets is far too much
    """
    split = n // 2
    seed = 1349565 // 101
    seed1 = 1349565 // 13
    seed2 = 1349565 // 11
    df = pickle.load(open(lstmdata + "pandas_tweets_sample_5_million.p", "rb"))
    if n < 5000000:
        df = pd.concat([df[df["class"] == "bot"].sample(n = split, random_state = seed1),df[df["class"] == "human"].sample(n = split, random_state = seed2)], ignore_index = True)
    df = df.sample(frac = 1.0, random_state = seed)
    
    return df

def check_counts(df):
    print(sum(df["tokenised_text"].str.count("<url>") == df["url_count"]) / len(df))
    print(sum(df["tokenised_text"].str.count("<hashtag>") == df["hashtag_count"]) / len(df))
    print(sum(df["tokenised_text"].str.count("<user>") == df["mention_count"]) / len(df))
    
    countdf = df["tokenised_text"].str.split().str.len()






def tensorfy(df, max_tokens = 50000, output_sequence_length = 40, standardize = True, batch_size = 128):
    """
    Note: (using df["tokenised_text"].str.split().str.len().quantile(q = 0.95) )
        90% of tweets have <30 tokens
        95% of tweets have <34 tokens
        98% of tweets have <43 tokens
        99% of tweets have <51 tokens
        
        97% of tweets have <40 tokens
    Note: (using df["tweet"].str.len().describe() )
        Average number of tokens is 16
    """
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length, standardize = None)
    text_to_adapt_to = tf.data.Dataset.from_tensor_slices(df["tokenised_text"].tolist()).batch(batch_size)
    vectorizer.adapt(text_to_adapt_to)
    
    # Vectorizing training data in batches
    vectorized_training_data_batches = []
    r = int(np.ceil(len(df)/batch_size))
    print(" ")
    for i in range(r):
        sys.stdout.write("\r" + str(i+1) + " out of " + str(r))
        vectorized_training_data_batches.append(vectorizer(df["tokenised_text"][batch_size*i:batch_size*(i+1)].tolist()))
    trn_tweet_vector = tf.concat(vectorized_training_data_batches, axis = 0)
    print("                           ")
    print("Finished Vectorizing")
    
    # Metadata - should we standardise this?
    trn_metadata = df[['hashtag_count', 'mention_count', 'url_count',
       'retweet_count', 'favorite_count']].to_numpy()
    
    if standardize:
        scaling = StandardScaler()
        trn_metadata = scaling.fit_transform(trn_metadata)
        # X_test = scaling.transform(X_test)
    
    # Retrieving labels
    trn_labels = df["class"].replace({"human" : 0, "bot" : 1})     
    
    return vectorizer, scaling, trn_tweet_vector, trn_metadata, trn_labels



def create_embeddings_index():
    """
    Loading the twitter glove file 100d (100 dimensions)
    *DIMENSIONS = 100, IF THIS WERE TO CHANGE, CHANGE THE IF STATEMENT*
    """
    embeddings_index = {}
    with open(glove_file,'r',encoding = "utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype = "f", sep=" ")
            if len(coefs) == 100:# checking to see it matches the dimension!
                embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index

def create_embedding_matrix(vectorizer, embeddings_index):
    """
    Creates the embedding_matrix
    NOTE: num_tokens = len(voc) + 2 (the +2 is for padding of "[UNK]" (unknown) and "" (empty space))
    NOTE: If dimensions change, change embedding_dim (currently set to 100)
    """
    hits = 0; misses = 0 # variables to record number of words converteds
    # retrieving voc and word_index from vectorizer
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    # Shape of the embedding matrix:
    num_tokens = len(voc) + 2
    embedding_dim = 100
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():# iterating through the items in the vectorizer
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix

# df = load_sample(5000000)
# vectorizer, scaling, trn_tweet_vector, trn_metadata, trn_labels = tensorfy(df)
# embeddings_index = create_embeddings_index()
# embedding_matrix = create_embedding_matrix(vectorizer, embeddings_index)

# vectorizer.get_vocabulary()[:10]

# 4. Model Structure ----------------------------------------------------------

def contextual_lstm_model(embed_mat, optimizer = "rmsprop"):
    """
    Defines LSTM model structure
    
    Add a dropout layer to avoid overfitting?
    """
    
    max_length = 40 # = output_sequence_length
    tweet_input = Input(shape=(max_length,), dtype = 'int64', name = "tweet_input")
    embed_layer = Embedding(50002, 100, embeddings_initializer=Constant(embed_mat), input_length = 40, trainable = False)(tweet_input)
    lstm_layer = LSTM(100)(embed_layer)
    auxiliary_output = Dense(1, activation="sigmoid", name = "auxiliary_output")(lstm_layer) # side output, won't contribute to the contextual LSTM, just so we can see what the LSTM is doing/have predicted
    metadata_input = Input(shape = (5,), name = "metadata_input") # there are 5 pieces of metadata
    new_input = Concatenate(axis=-1)([cast(lstm_layer, "float32"), cast(metadata_input, "float32") ])
    hidden_layer_1 = Dense(128, activation = "relu")(new_input)
    hidden_layer_2 = Dense(128, activation = "relu")(hidden_layer_1)
    final_output = Dense(1, activation = "sigmoid", name = "final_output")(hidden_layer_2)
    
    
    # compiling the model:
    model = Model(inputs = [tweet_input, metadata_input], outputs = [final_output, auxiliary_output])
    model.compile(optimizer = optimizer,
                 loss = "binary_crossentropy",
                 metrics = ["acc"],
                 loss_weights = [0.8, 0.2])
    model.summary()
    return model

# 5. Training Model -----------------------------------------------------------

def train_contextual_lstm_model(num_epochs = 10, optimizer = "Adam", batch_size = 128, load_weights_path = None, vectorizer = None, trn_tweet_vector = None, trn_metadata = None, trn_labels = None, embeddings_index = None, embedding_matrix = None):
    # ====================
    # Check point:
    # ====================
    # Model weights will be saved if they are the best seen so far (in terms of validation accuracy)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=weightsavepath, save_weights_only=True, # can set this to false but would then need to use load_model instead of load_weights
        save_freq = "epoch",
        monitor='final_output_acc',
        mode='max',
        save_best_only=True)
    
    
    # ====================
    # Loading the model structure:
    # ====================
    print("Loading model structure / weights")
    # model = mixed_contextual_lstm_model()
    model = contextual_lstm_model(embedding_matrix, optimizer)
    
    if load_weights_path != None:
        # ====================
        # loading previous model weights:
        # ====================
        model.load_weights(load_weights_path)
    
    
    # ====================
    # TRAINING
    # ====================
    print("Begin training...")
    # Training the model on very few epochs
    split = int(0.8*len(trn_tweet_vector))
    history = model.fit(
        {'tweet_input': trn_tweet_vector[:split,:], 'metadata_input': trn_metadata[:split,:]},
        {'final_output': trn_labels[:split], 'auxiliary_output': trn_labels[:split]},
        validation_data = (
            {'tweet_input': trn_tweet_vector[split:, :], 'metadata_input': trn_metadata[split:,:]},
            {'final_output': trn_labels[split:], 'auxiliary_output': trn_labels[split:]}
            ),
        epochs = num_epochs,
        batch_size = batch_size,
        callbacks=[model_checkpoint_callback]
        )
    return model, history

# model, history = train_contextual_lstm_model(num_epochs = 10, optimizer = "Adam", batch_size = 1024, load_weights_path = weightsavepath, vectorizer = vectorizer, trn_tweet_vector = trn_tweet_vector, trn_metadata = trn_metadata, trn_labels = trn_labels, embeddings_index = embeddings_index, embedding_matrix = embedding_matrix)

