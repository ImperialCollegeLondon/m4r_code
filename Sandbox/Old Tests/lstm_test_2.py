"""
LSTM bot detector trained on the italian dataset with contextual features
i.e. run LSTM with GloVe embeddings trained on italian tweets
then feed the output of the LSTM along with the contextual features (retweet count etc.)
into a NN or other.
"""


# Importing modules
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence # possibly not needed?
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, LSTM, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# File Paths
path_to_data = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"

# Importing and preprocessing the Italian dataset



