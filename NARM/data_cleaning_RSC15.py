# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:27:19 2024

@author: shefai
"""
import pandas as pd
import datetime as dt
import numpy as np

class data_cleaning_RSC15:
    def __init__(self, file, ratio = 64):
        
        data = pd.read_csv( file, sep=',')
        data.columns = ['SessionId', 'TimeStr', 'ItemId', 'Cate']
        data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
        del(data['TimeStr'])
        session_lengths = data.groupby('SessionId').size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
        item_supports = data.groupby('ItemId').size()
        data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
        session_lengths = data.groupby('SessionId').size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]

        print("...Data info after filtering...")
        print("Number of click:   "+str(len(data)))
        print("Number of sessions:   "+  str(  len(data["SessionId"].unique()) ))
        print("Number of items:   "+  str(  len(data["ItemId"].unique()) )) 
         
        # select latest ratio..... 
        latest_data = int(len(data) - len(data) / ratio)
        data = data.iloc[latest_data:, :]

        tmax = data.Time.max()
        session_max_times = data.groupby('SessionId').Time.max()
        session_train = session_max_times[session_max_times < tmax-86400].index
        session_test = session_max_times[session_max_times > tmax-86400].index
        train = data[np.in1d(data.SessionId, session_train)]
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
        

        print("...Info about testing data...")
        print("Number of click:   "+str(len(test)))
        print("Sessions:  "+str(len(test["SessionId"].unique())))
        print("Items:  "+str(len(test["ItemId"].unique())))
        


        print("...Data preparation for DL model")
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
        self.word2index = word2index


        

        
























        
        
