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

# 2. File Paths ---------------------------------------------------------------
# CHANGE ALL OF THESE TO WHERE THEY ARE STORED:
m4r_data ="C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"
# GLOVE FILES CAN BE OBTAINED FROM: https://nlp.stanford.edu/data/glove.twitter.27B.zip
glove_file_50 = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\glove.twitter.27B\\glove.twitter.27B.50d.txt"
glove_file_100 = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\glove.twitter.27B\\glove.twitter.27B.100d.txt"
weightsavepath = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_code\\Tweet Level Detection\\contextual_LSTM_weights_balanced\\"
figure_path = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Report\\Figures\\"



# 1. Packages -----------------------------------------------------------------
import sys, pickle, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
sns.set(font="Arial")


# 3. Preprocessing ------------------------------------------------------------ (DONE)
def balance_data(save = False):
    """
    Balances and saves the tweet data (from all available datasets)
    """
    # Load all available training data
    df = pickle.load(open(m4r_data + "tweet_training_data.p", "rb"))
    df = df.sample(frac = 1.0, random_state = 1349565 // 17).reset_index(drop = True) # shuffle
    df["user.id"] = df["user.id"].astype("int64")
    df = df.groupby("user.id").head(200) # pick a random sample of up to 200 tweets from each person
    
    # Pring and show the balance of the data:
    print("Bots:  ", sum(df["class"] == "bot") / len(df) * 100, "%")
    print("Humans:", sum(df["class"] == "human") / len(df) * 100, "%")
    ax = sns.histplot(data = df, x = "dataset"); ax.set_title("Balance of Datasets");  plt.xticks(rotation = 60); plt.show()
    
    if save:
        pickle.dump(df, open(m4r_data + "balanced_tweet_training_data.p" , "wb"))
    
def get_tweet_data():
    """
    Retrieves BALANCED training dataset
    """
    df = pickle.load(open(m4r_data + "balanced_tweet_training_data.p" , "rb"))
    df = df.sample(frac = 1.0, random_state = 1349565 // 19).reset_index(drop = True)
    return df

def tensorfy(df, max_tokens = 50000, output_sequence_length = 40, batch_size = 2048):
    """
    Vectorizes text, standardizes metadata
    """
    # Train-Val split (during training, the train data will be further split into train-test as well, but we need two separate datasets to get F1, Recall, Precision etc...)
    split = int(len(df) * 0.8)
    
    
    # "Training the vectorizer"
    print("Adapting vectorizer")
    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length, standardize = None)
    text_to_adapt_to = tf.data.Dataset.from_tensor_slices(df.iloc[:split,:]["tokenised_text"].tolist()).batch(batch_size)
    vectorizer.adapt(text_to_adapt_to)
    
    # Vectorizing the tweets in batches
    vectorized_data_batches = []
    r = int(np.ceil(len(df)/batch_size))
    print(" ")
    for i in range(r):
        sys.stdout.write("\r" + str(i+1) + " out of " + str(r))
        vectorized_data_batches.append(vectorizer(df["tokenised_text"][batch_size*i:batch_size*(i+1)].tolist()))
    tweet_vector = tf.concat(vectorized_data_batches, axis = 0)
    print("                           ")
    print("Finished Vectorizing")
    
    # Metadata - should we standardise this?
    metadata = df[['hashtag_count', 'mention_count', 'url_count',
       'retweet_count', 'favorite_count']].to_numpy()
    
    # Standardizing the metadata
    scaling = StandardScaler()
    metadata_vector = scaling.fit_transform(metadata)
    
    # Retrieving labels
    label_vector = df["class"].replace({"human" : 0, "bot" : 1}).to_numpy()
    
    return vectorizer, scaling, tweet_vector, metadata_vector, label_vector
    

def get_embeddings_index_100():
    """
    Loading the twitter glove file 100d (100 dimensions)
    *DIMENSIONS = 100
    """
    embeddings_index = {}
    with open(glove_file_100,'r',encoding = "utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype = "f", sep=" ")
            if len(coefs) == 100: # checking to see it matches the dimension!
                embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index

def get_embeddings_index_50():
    """
    Loading the twitter glove file 50d (50 dimensions)
    *DIMENSIONS = 50
    """
    embeddings_index = {}
    with open(glove_file_50,'r',encoding = "utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, dtype = "f", sep=" ")
            if len(coefs) == 50:# checking to see it matches the dimension!
                embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index

def get_embedding_matrix(vectorizer, embeddings_index, embedding_dim = 100):
    """
    Creates the embedding_matrix
    NOTE: num_tokens = len(voc) + 2 (the +2 is for padding of "[UNK]" (unknown) and "" (empty space))
    """
    hits = 0; misses = 0 # variables to record number of words converteds
    # retrieving voc and word_index from vectorizer
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    # Shape of the embedding matrix:
    num_tokens = len(voc) + 2
    # embedding_dim = 100
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():# iterating through the items in the vectorizer
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV" (out of vocabulary)
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    return embedding_matrix
    
def run_preprocessing(max_tokens = 50000, embedding_dim = 100, output_sequence_length = 40):
    """
    Runs the tensorfy function (above)
    
    """
    df = get_tweet_data() # Retrieving training dataset
    # Running Tensorfy (vectorizes tweets, standardizes metadata)
    vectorizer, scaling, tweets, metadata, labels= tensorfy(df, max_tokens = max_tokens, output_sequence_length = output_sequence_length, batch_size = 2048)
    # Retrieve embeddings index
    if embedding_dim == 100:
        embeddings_index = get_embeddings_index_100()
    elif embedding_dim == 50:
        embeddings_index = get_embeddings_index_50()
    embedding_matrix = get_embedding_matrix(vectorizer, embeddings_index, embedding_dim)

    # vectorizer.get_vocabulary()[:10] # this would pring the top 10 most occurring words in our training dataset
    
    return vectorizer, scaling, tweets, metadata, labels, embeddings_index, embedding_matrix

# 4. Model Structure ----------------------------------------------------------
def contextual_lstm_model(embed_mat, optimizer = "Adam"):
    """
    Defines LSTM model structure:
    """
    max_length = 40 # = output_sequence_length # maximum number of tokens we will allow (by truncating and padding tweets)
    tweet_input = Input(shape=(max_length,), dtype = 'int64', name = "tweet_input")
    # New embedding layer: with masking
    embed_layer = Embedding(50002, 100, embeddings_initializer=Constant(embed_mat), input_length = 40, trainable = False, mask_zero = True)(tweet_input)
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

def contextual_lstm_model_50(embed_mat, embed_size = 30000, embed_dim = 50, max_length = 40, optimizer = "Adam"):
    """
    Defines LSTM model structure: using Dimension 50 word vectors
    """
     # = output_sequence_length
    tweet_input = Input(shape=(max_length,), dtype = 'int64', name = "tweet_input")
    # New embedding layer: with masking
    embed_layer = Embedding(input_dim = embed_size, output_dim = embed_dim, embeddings_initializer=Constant(embed_mat), input_length = max_length, trainable = False, mask_zero = True)(tweet_input)
    lstm_layer = LSTM(embed_dim)(embed_layer)
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
def train_contextual_lstm_model(num_epochs = 10, optimizer = "Adam", batch_size = 256, load_weights_path = None, vectorizer = None, tweets = None, metadata = None, labels = None, embeddings_index = None, embedding_matrix = None, save_to_path = weightsavepath, moddd = 0):
    # ====================
    # Check point:
    # ====================
    # Model weights will be saved if they are the best seen so far (in terms of validation accuracy)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = save_to_path, save_weights_only=True, # can set this to false but would then need to use load_model instead of load_weights
        save_freq = "epoch",
        monitor='final_output_acc',
        mode='max',
        save_best_only=True)
    
    # ====================
    # Loading the model structure:
    # ====================
    print("Loading model structure / weights")
    # model = mixed_contextual_lstm_model()
    
    if moddd == 0:  # usual model, dimension 100, 50002 word vectors, max 40 tweet length etc...
        model = contextual_lstm_model(embedding_matrix, optimizer)
    elif moddd == 1:
        model = contextual_lstm_model_50(embedding_matrix, embed_size = 30000, embed_dim = 50, max_length = 40, optimizer = "Adam")
    
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
    
    split = int(0.8*len(tweets)) # training testing split is 80:20
    
    history = model.fit(
        {'tweet_input': tweets[:split,:], 'metadata_input': metadata[:split,:]},
        {'final_output': labels[:split], 'auxiliary_output': labels[:split]},
        validation_data = (
            {'tweet_input': tweets[split:, :], 'metadata_input': metadata[split:,:]},
            {'final_output': labels[split:], 'auxiliary_output': labels[split:]}
            ),
        epochs = num_epochs,
        batch_size = batch_size,
        callbacks=[model_checkpoint_callback]
        )
    return model, history


def run_training(epochs = 30, batch = 1024):
    """
    Trains model
    epochs = #epochs to train
    batch = training batch size
    """
    
    vectorizer, scaling, tweets, metadata, labels, embeddings_index, embedding_matrix = run_preprocessing()
    
    model, history = train_contextual_lstm_model(num_epochs = epochs, optimizer = "Adam", batch_size = batch, vectorizer = vectorizer, tweets = tweets, metadata = metadata, labels = labels, embeddings_index = embeddings_index, embedding_matrix = embedding_matrix, save_to_path = weightsavepath)






def get_scores_5(y_trn, p_trn, y_tst, p_tst):
    """
    Retrieves the scores of the classifier clf
    """
    scores = []
    
    # Appending training score:
    scores += [accuracy_score(y_trn, p_trn)]
    
    # Appending testing scores:
    scores += [accuracy_score(y_tst, p_tst)]
    scores += [precision_score(y_tst, p_tst)]
    scores += [recall_score(y_tst, p_tst)]
    scores += [f1_score(y_tst, p_tst)]
    
    return scores

# Load Model:
def load_trained_model(optimizer = "Adam", save = False):
    # Running Contextual LSTM on training and testing data:
    # Takes about 8 minutes to run LSTM to predict on the training set
    vectorizer, scaling, tweets, metadata, labels, embeddings_index, embedding_matrix = run_preprocessing()
    model = contextual_lstm_model(embedding_matrix)
    model.load_weights(weightsavepath)
    split = int(0.8*len(tweets))
    trn_predictions = model.predict({"tweet_input" : tweets[:split,:], 'metadata_input' : metadata[:split,:]})
    tst_predictions = model.predict({"tweet_input" : tweets[split:,:], 'metadata_input' : metadata[split:,:]})
    
    trn_predictions_lstm = np.round((trn_predictions[1]).reshape(-1,))
    trn_predictions_contextual_lstm = np.round((trn_predictions[0]).reshape(-1))
    
    tst_predictions_lstm = np.round((tst_predictions[1]).reshape(-1,))
    tst_predictions_contextual_lstm = np.round((tst_predictions[0]).reshape(-1))
    
    
    # Retrieving Scores for the classifiers
    
    criterion_names = ["Train Accuracy", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"]

    score_dataframe = pd.DataFrame()

    
    # Contextual LSTM Scores
    
    sf = pd.DataFrame(columns = ["Model", "Criterion" ,"Score"])
    sf["Model"] = ["Contextual LSTM"]*5
    sf["Criterion"] = criterion_names
    sf["Score"] = get_scores_5(labels[:split], trn_predictions_contextual_lstm, labels[split:], tst_predictions_contextual_lstm)
    
    score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
    
    # LSTM Scores

    sf = pd.DataFrame(columns = ["Model", "Criterion" ,"Score"])
    sf["Model"] = ["LSTM"]*5
    sf["Criterion"] = criterion_names
    sf["Score"] = get_scores_5(labels[:split], trn_predictions_lstm, labels[split:], tst_predictions_lstm)
    
    score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)

    
    # Other Models...
    # RFC
    # clf = RandomForestClassifier(random_state = 25, max_depth = 100)
    # clf.fit(metadata[:split], labels[:split])
    # p_trn = np.round(clf.predict(metadata[:split]))
    # p_tst = np.round(clf.predict(metadata[split:]))
    # sf = pd.DataFrame(columns = ["Model", "Criterion" ,"Score"])
    # sf["Model"] = ["RFC"]*5; sf["Criterion"] = criterion_names
    # sf["Score"] = get_scores_5(labels[:split], p_trn, labels[split:], p_tst)
    # score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
    
    # LR
    clf  = LogisticRegression(max_iter = 10000, random_state=25, penalty='l2')
    clf.fit(metadata[:split], labels[:split])
    p_trn = np.round(clf.predict(metadata[:split]))
    p_tst = np.round(clf.predict(metadata[split:]))
    sf = pd.DataFrame(columns = ["Model", "Criterion" ,"Score"])
    sf["Model"] = ["LR"]*5; sf["Criterion"] = criterion_names
    sf["Score"] = get_scores_5(labels[:split], p_trn, labels[split:], p_tst)
    score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
    
    # SVM
    clf = SGDClassifier(max_iter=1000, tol=1e-8, random_state = 25)
    clf.fit(metadata[:split], labels[:split])
    p_trn = np.round(clf.predict(metadata[:split]))
    p_tst = np.round(clf.predict(metadata[split:]))
    sf = pd.DataFrame(columns = ["Model", "Criterion" ,"Score"])
    sf["Model"] = ["SVM"]*5; sf["Criterion"] = criterion_names
    sf["Score"] = get_scores_5(labels[:split], p_trn, labels[split:], p_tst)
    score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)
    
    # AB
    clf  = AdaBoostClassifier(n_estimators = 50, random_state = 25)
    clf.fit(metadata[:split], labels[:split])
    p_trn = np.round(clf.predict(metadata[:split]))
    p_tst = np.round(clf.predict(metadata[split:]))
    sf = pd.DataFrame(columns = ["Model", "Criterion" ,"Score"])
    sf["Model"] = ["AB"]*5; sf["Criterion"] = criterion_names
    sf["Score"] = get_scores_5(labels[:split], p_trn, labels[split:], p_tst)
    score_dataframe = pd.concat([score_dataframe, sf], ignore_index = True)

    # Plotting

    Title = "Comparison of Tweet Level Detection Models\n"
    Title += "(trained on all datasets with 80:20 split)"
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
    plt.ylim(0,1)
    if save:
        plt.savefig(figure_path + "tweet_lvl_training_compare_models_all.pdf", bbox_inches = "tight")
    plt.show()