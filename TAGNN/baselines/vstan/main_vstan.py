# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024

@author: shefai
"""
from TAGNN.data_preprocessing.digi_data_preprocessing import *;
from TAGNN.data_preprocessing.rsc15_data_preprocessing import *;

from TAGNN.baselines.vstan.vstan  import *
from pathlib import Path
root_path = '\\'
from TAGNN.accuracy_measures import *


class VSTAN_MAIN:
    
    def __init__(self, data_path, result_path, dataset = "diginetica"):
        self.dataset = dataset
        self.result_path = result_path

        if dataset == "diginetica":
            self.k = 2000
            self.sample_size = 5500
            self.lambda_spw = 0.104
            self.lambda_snh = 53
            self.lambda_inh = 2.4
            self.lambda_idf = 1
            
            name = "train-item-views.csv"
            data_ = load_data(data_path / name) 


            data_ = filter_data(data_)
            self.train_data, self.test_data, self.unique_items_ids = split_data_digi_knn(data_)
            
            
        elif dataset == 'yoochoose1_64' or dataset == 'yoochoose1_4':
            
            self.k = 1530
            self.sample_size = 5600
            self.lambda_spw = 0.98
            self.lambda_snh = 300
            self.lambda_inh = 2.4
            self.lambda_idf = 6
            
        
            name = "yoochoose-clicks.dat"
            data_ = load_data_rsc15(data_path / name) 


            data_ = filter_data_rsc15(data_)
            self.train_data, self.test_data, self.unique_items_ids = split_data_rsc15_knn(data_)
            
        else:
            print("Mention your datatypes")
            
            
    def fit_(self, mrr, hr):
        
        obj1 = VSKNN_STAN(k = self.k,  sample_size = self.sample_size, lambda_spw = self.lambda_spw, lambda_snh = self.lambda_snh, lambda_inh = self.lambda_inh, lambda_idf = self.lambda_idf )
        obj1.fit(self.train_data)
        
        session_key ='SessionId'
        time_key='Time'
        item_key= 'ItemId'
        
        # Intialize accuracy measures.....
        MRR_dictionary = dict()
        for i in mrr:
            MRR_dictionary["MRR_"+str(i)] = MRR(i)
            
        HR_dictionary = dict()
        for i in hr:
            HR_dictionary["HR_"+str(i)] = HR(i)
        
        test_data = self.test_data
        test_data.sort_values([session_key, time_key], inplace=True)
        items_to_predict = self.unique_items_ids
        
        # Previous item id and session id....
        prev_iid, prev_sid = -1, -1
        
        
        for i in range(len(test_data)):
            sid = test_data[session_key].values[i]
            iid = test_data[item_key].values[i]
            ts = test_data[time_key].values[i]
            if prev_sid != sid:
                # this will be called when there is a change of session....
                prev_sid = sid
            else:
                # prediction starts from here.......
                preds = obj1.predict_next(sid, prev_iid, items_to_predict, ts)
                preds[np.isnan(preds)] = 0
    #             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                preds.sort_values( ascending=False, inplace=True )    
    
                for key in MRR_dictionary:
                    MRR_dictionary[key].add(preds, iid)
                    
                
                # Calculate the HR values
                for key in HR_dictionary:
                    HR_dictionary[key].add(preds, iid)
                
                
                
                
            prev_iid = iid
            
        # get the results of MRR values.....
        result_frame = pd.DataFrame()    
        for key in MRR_dictionary:
            print(key +"   "+ str(  MRR_dictionary[key].score()    ))
            result_frame[key] =   [MRR_dictionary[key].score()]
           
        # get the results of MRR values.....    
        for key in HR_dictionary:
            print(key +"   "+ str(  HR_dictionary[key].score()    ))
            result_frame[key] = [HR_dictionary[key].score()]
        
        name = "TAGNN_VSTAN_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path /name, sep = "\t", index = False) 
        
       
        
        
        
        
        


