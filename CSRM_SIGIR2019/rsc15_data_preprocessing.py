# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:25:25 2024

@author: shefai
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import csv
import time
import pickle
import operator
from datetime import datetime
import random

COLS = [0, 2, 3, 4]
# days test default config
DAYS_TEST = 1
MINIMUM_ITEM_SUPPORT = 5
MINIMUM_SESSION_LENGTH = 2

# preprocessing from original gru4rec -  uses just the last day as test
    
def load_data_rsc15( file ):     
    #load csv
    
    data = pd.read_csv( file, sep=',', header=None, usecols=[0,1,2])
    #specify header names
    data.columns = ['SessionId', 'TimeStr', 'ItemId']
    data['Time'] = data.TimeStr.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
    del(data['TimeStr'])
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc)
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    print('Loaded data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    return data

def filter_data_rsc15( data, ratio, min_item_support=MINIMUM_ITEM_SUPPORT, min_session_length=MINIMUM_SESSION_LENGTH ):


    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths >= min_session_length ].index)]
    #filter item support
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[ item_supports>= min_item_support ].index)]
    #fiter session length
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[ session_lengths >= min_session_length ].index)]

    # select the latest fractions of 
    
    #output
    data_start = datetime.fromtimestamp( data.Time.min(), timezone.utc )
    data_end = datetime.fromtimestamp( data.Time.max(), timezone.utc )
    
    latest_data = int(len(data) - len(data) / ratio)
    data = data.iloc[latest_data:, :]
    print('Filtered data set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tSpan: {} / {}\n\n'.
          format( len(data), data.SessionId.nunique(), data.ItemId.nunique(), data_start.date().isoformat(), data_end.date().isoformat() ) )
    return data

def split_data_rsc15( data, tes_days = DAYS_TEST):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    # Last day data is used  for testing of models.
    session_train = session_max_times[session_max_times < tmax-(86400 *tes_days)].index
    session_test = session_max_times[session_max_times >= tmax-(86400 *tes_days)].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId', 'Time'],  inplace = True)
    min_date = datetime.fromtimestamp(test.Time.min())
    max_date = datetime.fromtimestamp(test.Time.max())
    
    ########### Training and testing data
    difference = max_date - min_date
    print("Number of testing days:", difference.days)
    print("Info about training data")
    print("Number of clicks : %d, Number of sessions: %d, Number of items: %d" % (len(train), len( train["SessionId"].unique() ), len( train["ItemId"].unique() ))) 
    print("Info about test data")
    print("Number of clicks : %d, Number of sessions: %d, Number of items: %d" % (len(test), len( test["SessionId"].unique() ), len( test["ItemId"].unique() ))) 
    # create data strucutre for GNN models
    session_key = "SessionId"
    item_key = "ItemId"
    index_session = train.columns.get_loc( session_key)
    index_item = train.columns.get_loc( item_key )
    session_item_train = {}
    # Convert the session data into sequence
    for row in train.itertuples(index=False):
        if row[index_session] in session_item_train:
            session_item_train[row[index_session]] += [(row[index_item])] 
        else: 
            session_item_train[row[index_session]] = [(row[index_item])]
    word2index ={}
    index2wiord = {}
    item_no = 1
    for key, values in session_item_train.items():
        length = len(session_item_train[key])
        for i in range(length):
            if session_item_train[key][i] in word2index:
                session_item_train[key][i] = word2index[session_item_train[key][i]]
            else:
                word2index[session_item_train[key][i]] = item_no
                index2wiord[item_no] = session_item_train[key][i]
                session_item_train[key][i] = item_no
                item_no +=1
    
    all_sessionsOfinTrianingData = list(session_item_train.keys())
    SessionforValidationData = int(0.1 * len(all_sessionsOfinTrianingData))
    valid_te_list = random.sample(all_sessionsOfinTrianingData, SessionforValidationData)

    features_train = []
    targets_train = []
    for key, value in session_item_train.items():
        for i in range(1, len(value)):
            targets_train.append(value[-i])
            features_train.append(value[:-i])

    session_item_test = {}
    # Convert the session data into sequence
    for row in test.itertuples(index=False):
        if row[index_session] in session_item_test:
            session_item_test[row[index_session]] += [(row[index_item])] 
        else: 
            session_item_test[row[index_session]] = [(row[index_item])]
            
    for key, values in session_item_test.items():
        length = len(session_item_test[key])
        for i in range(length):
            if session_item_test[key][i] in word2index:
                session_item_test[key][i] = word2index[session_item_test[key][i]]
            else:
                word2index[session_item_test[key][i]] = item_no
                index2wiord[item_no] = session_item_test[key][i]
                session_item_test[key][i] = item_no
                item_no +=1
    
    features_test = []
    targets_test = []
    for value in session_item_test.values():
        targets_test.append(value[-1])
        features_test.append(value[:-1])
    item_no = item_no +1
    
    ############ tuning... validation data
    total_number_sessions = list(train["SessionId"].unique())
    valid_te_list = int(0.1 * len(total_number_sessions))
    valid_te_list = random.sample(total_number_sessions, valid_te_list)
    valid_tr_list = [item for item in total_number_sessions if item not in valid_te_list]

    # validation testing data............ valid_tr
    valid_te = train[train["SessionId"].isin(valid_te_list)]
    valid_tr_dict = valid_te.groupby('SessionId')['ItemId'].apply(list).to_dict()
    for key, values in valid_tr_dict.items():
        length = len(valid_tr_dict[key])
        for i in range(length):
            valid_tr_dict[key][i] = word2index[valid_tr_dict[key][i]]
    valid_features_test = []
    valid_targets_test = []        
    for value in valid_tr_dict.values():
        valid_targets_test.append(value[-1])
        valid_features_test.append(value[:-1])
    ######### validation training data
    valid_tr = train[train["SessionId"].isin(valid_tr_list)]
    valid_tr_dict = valid_tr.groupby('SessionId')['ItemId'].apply(list).to_dict() # train_tr_dict
    
    for key, values in valid_tr_dict.items():
        length = len(valid_tr_dict[key])
        for i in range(length):
            valid_tr_dict[key][i] = word2index[valid_tr_dict[key][i]]
    valid_features_train = []
    valid_targets_train = []        
    for value in valid_tr_dict.values():
        valid_targets_train.append(value[-1])
        valid_features_train.append(value[:-1])

    return [features_train, targets_train], [features_test, targets_test], item_no, [valid_features_train, valid_targets_train], [valid_features_test, valid_targets_test], train, test, valid_tr, valid_te

def split_data_temp(data, tes_days = DAYS_TEST):
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    # Last day data is used  for testing of models.
    session_train = session_max_times[session_max_times < tmax-(86400 *tes_days)].index
    session_test = session_max_times[session_max_times >= tmax-(86400 *tes_days)].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    train.sort_values(by=['SessionId', 'Time'],  inplace = True)
    test.sort_values(by=['SessionId', 'Time'],  inplace = True)
    return train, test
    


def split_data_rsc15_baseline( data):
    train, test = split_data_temp(data)
    
    #train.to_csv("rsc15_train_full.txt", sep = "\t", index = False)
    #test.to_csv("rsc15_test.txt", sep = "\t", index = False)
    
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    
    
    difference = datetime.fromtimestamp(train.Time.max()) - datetime.fromtimestamp(train.Time.min())
    print("Number of training days:", difference.days)
    
    difference = datetime.fromtimestamp(test.Time.max()) - datetime.fromtimestamp(test.Time.min())
    print("Number of test days:", difference.days)
    
    
    train_validation, test_validation = split_data_temp(train)
    
    #train_validation.to_csv("rsc15_train_tr.txt", sep = "\t", index = False)
    #test_validation.to_csv("rsc15_train_valid.txt", sep = "\t", index = False)
    
    unique_items_ids = data["ItemId"].unique()
    
    session_key = "SessionId"
    item_key = "ItemId"
    index_session = train.columns.get_loc( session_key)
    index_item = train.columns.get_loc( item_key )
    
    
    
            
            
    session_item_test = {}
    # Convert the session data into sequence
    for row in test.itertuples(index=False):
        if row[index_session] in session_item_test:
            session_item_test[row[index_session]] += [(row[index_item])] 
        else: 
            session_item_test[row[index_session]] = [(row[index_item])]
            
    
    features_test = []
    targets_test = []
    for value in session_item_test.values():
        for i in range(1, len(value)):
            targets_test.append(value[-i])
            features_test.append(value[:-i])
    
    return train, [features_test, targets_test], unique_items_ids
   
def split_data_rsc15_knn( data):
    train, test = split_data_temp(data)
    
    train.to_csv("rsc15_train_full.txt", sep = "\t", index = False)
    test.to_csv("rsc15_test.txt", sep = "\t", index = False)
    
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(),
                                                                             train.ItemId.nunique()))
    #train.to_csv(output_file + 'train.txt', sep='\t', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(),
                                                                       test.ItemId.nunique()))
    
    
    difference = datetime.fromtimestamp(train.Time.max()) - datetime.fromtimestamp(train.Time.min())
    print("Number of training days:", difference.days)
    
    difference = datetime.fromtimestamp(test.Time.max()) - datetime.fromtimestamp(test.Time.min())
    print("Number of test days:", difference.days)
    
    
    train_validation, test_validation = split_data_temp(train)
    
    train_validation.to_csv("rsc15_train_tr.txt", sep = "\t", index = False)
    test_validation.to_csv("rsc15_train_valid.txt", sep = "\t", index = False)
    
    unique_items_ids = data["ItemId"].unique()
    
    session_key = "SessionId"
    item_key = "ItemId"
    index_session = train.columns.get_loc( session_key)
    index_item = train.columns.get_loc( item_key )
    
    
    return train, test, unique_items_ids

#
# if __name__ == '__main__':
#     path = "datasets/rsc15/yoochoose-clicks"
    
#     dataset = load_data_rsc15(path)
#     filter_data = filter_data_rsc15(dataset)
#     train, test = split_data_rsc15(filter_data)
#     #features_train, targets_train, features_test, targets_test, item_no = split_data(filter_data)




