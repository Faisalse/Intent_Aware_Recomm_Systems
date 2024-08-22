# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:27:19 2024

@author: shefai
"""

import pandas as pd
import numpy as np
from datetime import datetime

class data_cleaning_DIGI:

    def __init__(self, file):
        data = pd.read_csv(file, sep =";")
        del data["userId"]
        del data['timeframe']
        data['Time'] = data['eventdate'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timestamp())
        del data['eventdate']

        session_lengths = data.groupby('sessionId').size()
        data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]
        item_supports = data.groupby('itemId').size()
        data = data[np.in1d(data.itemId, item_supports[item_supports>=5].index)]
        session_lengths = data.groupby('sessionId').size()
        data = data[np.in1d(data.sessionId, session_lengths[session_lengths>1].index)]
        data.rename(columns = {"sessionId":"SessionId", 'itemId':'ItemId'}, inplace = True) 
        print("...Data info after filtering...")
        print("Number of clicks:   "+str(len(data)))
        print("Number of sessions:   "+  str(  len(data["SessionId"].unique()) ))
        print("Number of items:   "+  str(  len(data["ItemId"].unique()) ))

        tmax = data.Time.max()
        session_max_times = data.groupby('SessionId').Time.max()
        session_train = session_max_times[session_max_times < tmax-86400*7].index
        session_test = session_max_times[session_max_times > tmax-86400*7].index

        train = data[np.in1d(data.SessionId, session_train)]
        trlength = train.groupby('SessionId').size()
        train = train[np.in1d(train.SessionId, trlength[trlength>=2].index)]
        test = data[np.in1d(data.SessionId, session_test)]
        test = test[np.in1d(test.ItemId, train.ItemId)]
        tslength = test.groupby('SessionId').size()
        test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]

        train.sort_values(by=['SessionId', 'Time'],  inplace = True)
        test.sort_values(by=['SessionId', 'Time'],  inplace = True)


        print("...Info about training data...")
        print("Number of click:   "+str(len(train)))
        print("Sessions:  "+str(len(train["SessionId"].unique())))
        print("Items:  "+str(len(train["ItemId"].unique())))

        print("Min date:  "+     str(datetime.fromtimestamp(float(train["Time"].min()))))
        print("Max date:  "+     str(datetime.fromtimestamp(float(train["Time"].max()))))

        print("...Info about testing data...")
        print("Number of click:   "+str(len(test)))
        print("Sessions:  "+str(len(test["SessionId"].unique())))
        print("Items:  "+str(len(test["ItemId"].unique())))

        print("Min date:  "+     str(datetime.fromtimestamp(float(test["Time"].min()))))
        print("Max date:  "+     str(datetime.fromtimestamp(float(test["Time"].max()))))
        
        train_seq = train.groupby('SessionId')['ItemId'].apply(list).to_dict()
        
        word2index ={}
        index2word = {}
        item_no = 1
        
        for key, values in train_seq.items():
            length = len(train_seq[key])
            for i in range(length):
                if train_seq[key][i] in word2index:
                    train_seq[key][i] = word2index[train_seq[key][i]]
                    
                else:
                    word2index[train_seq[key][i]] = item_no
                    index2word[item_no] = train_seq[key][i]
                    train_seq[key][i] = item_no
                    item_no +=1
        self.train_seq_f = list()
        self.train_label = list()
        for key, seq_ in train_seq.items():
            self.train_seq_f.append(seq_[:-1])
            self.train_label.append(seq_[-1])
        # test data
        test_seq = test.groupby('SessionId')['ItemId'].apply(list).to_dict()
        for key, values in test_seq.items():
            length = len(test_seq[key])
            for i in range(length):
                if test_seq[key][i] in word2index:
                    test_seq[key][i] = word2index[test_seq[key][i]]
                    
        self.test_seq_f = list()
        self.test_label = list()
        for key, seq_ in test_seq.items():
            for i_ in range(1, len(seq_)):
                self.test_label.append(seq_[-i_])
                self.test_seq_f.append(seq_[:-i_])
        print("****************   "+str(len(self.test_seq_f)))
        self.word2index = word2index


        

        
























        
        
