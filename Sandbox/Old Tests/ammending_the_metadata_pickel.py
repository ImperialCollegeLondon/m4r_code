# -*- coding: utf-8 -*-
"""
Ammending the metadata pickle file - the 6 last values are strings not integers! or floats!
"""

import sys
import numpy as np
import pickle

path_to_italy = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\Italian Dataset\\datasets_full.csv\\datasets_full.csv\\"
path_to_m4rdata = "C:\\Users\\fangr\\Documents\\Year 4\\M4R\\m4r_data\\"


def second_ammendment():
    train_metadata = pickle.load(open(path_to_m4rdata+"italian_train_metadata.p", "rb" ))
    val_metadata = pickle.load(open(path_to_m4rdata+"italian_val_metadata.p", "rb" ))
    new_train_metadata = []
    new_val_metadata = []
    for i, vec in enumerate(train_metadata):
        vec_t_a = []
        vec_t_a += vec[:6]
        new_train_metadata.append(vec_t_a)
    for i, vec in enumerate(val_metadata):
        vec_t_a = []
        vec_t_a += vec[:6]
        new_val_metadata.append(vec_t_a)
    pickle.dump(new_train_metadata, open(path_to_m4rdata+"italian_train_metadata.p", "wb" ))
    pickle.dump(new_val_metadata, open(path_to_m4rdata+"italian_val_metadata.p", "wb" ))
    
    

def convert_string_metadata_to_number():
    train_metadata = pickle.load(open(path_to_m4rdata+"italian_train_metadata.p", "rb" ))
    val_metadata = pickle.load(open(path_to_m4rdata+"italian_val_metadata.p", "rb" ))
    new_train_metadata = []
    new_val_metadata = []
    new_train_metadata_w_time = []
    new_val_metadata_w_time = []
    for i, vec in enumerate(train_metadata):
        vec_to_add = []
        vec_to_add += [int(x) for x in vec[-6:]]
        new_train_metadata.append(vec_to_add)
        vec_to_add += vec[:6]
        new_train_metadata_w_time.append(vec_to_add)
    
    for i, vec in enumerate(val_metadata):
        vec_to_add = []
        vec_to_add += [int(x) for x in vec[-6:]]
        new_val_metadata.append(vec_to_add)
        vec_to_add += vec[:6]
        new_val_metadata_w_time.append(vec_to_add)
        
    # Overwriting the Italian dataset:
    pickle.dump(new_train_metadata, open(path_to_m4rdata+"italian_train_metadata.p", "wb" ))
    pickle.dump(new_val_metadata, open(path_to_m4rdata+"italian_val_metadata.p", "wb" ))
    pickle.dump(new_train_metadata_w_time, open(path_to_m4rdata+"italian_train_metadata_w_time.p", "wb" ))
    pickle.dump(new_val_metadata_w_time, open(path_to_m4rdata+"italian_val_metadata_w_time.p", "wb" ))
