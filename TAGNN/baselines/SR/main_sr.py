# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024

@author: shefai
"""
from TAGNN.data_preprocessing.digi_data_preprocessing import *;
from TAGNN.data_preprocessing.rsc15_data_preprocessing import *;

from TAGNN.baselines.SR.sr  import *
from pathlib import Path
root_path = '\\'
from TAGNN.accuracy_measures import *

class SequentialRulesMain:
    
    def __init__(self, data_path, result_path, dataset = "diginetica"):
        self.dataset = dataset
        self.result_path = result_path
        if dataset == "diginetica":
            self.steps = 12
            self.weighting = "quadratic"
            self.pruning = 20
            self.session_weighting = "div" 
            
            name = "train-item-views.csv"
            data_ = load_data(data_path / name) 
            data_ = filter_data(data_)
            self.train_data, self.test_data, self.unique_items_ids = split_data_digi_baseline(data_)
        elif dataset == 'yoochoose1_64' or dataset == 'yoochoose1_4':
    
            self.steps = 5
            self.weighting = "linear"
            self.pruning = 20
            self.session_weighting = "div"
            name = "yoochoose-clicks.dat"
            data_ = load_data_rsc15(data_path / name)
            data_ = filter_data_rsc15(data_)
            self.train_data, self.test_data, self.unique_items_ids = split_data_rsc15_baseline(data_)
        else:
            print("Mention your datatypes")
            
            
    def fit_(self, mrr, hr):
        
        obj1 = SequentialRules(steps = self.steps, weighting = self.weighting, pruning = self.pruning, session_weighting = self.session_weighting)
        obj1.fit(self.train_data)
        
        test_data = self.test_data[0]
        targets = self.test_data[1]
        
        
        # Intialize accuracy measures.....
        MRR_dictionary = dict()
        for i in mrr:
            MRR_dictionary["MRR_"+str(i)] = MRR(i)
            
        
        HR_dictionary = dict()
        for i in hr:
            HR_dictionary["HR_"+str(i)] = HR(i)
        
        
        
        print(len(self.unique_items_ids))
        for i in range(len(test_data)):
            
            last_inter_ = test_data[i][-1]
            tar_ = targets[i]
            
            predition_series = obj1.predict_next(last_inter_, self.unique_items_ids)
            
            
            for key in MRR_dictionary:
                MRR_dictionary[key].add(predition_series, tar_)
                
            
            # Calculate the HR values
            for key in HR_dictionary:
                HR_dictionary[key].add(predition_series, tar_)
            
            
            
        
        # get the results of MRR values.....
        result_frame = pd.DataFrame()    
        for key in MRR_dictionary:
            print(key +"   "+ str(  MRR_dictionary[key].score()    ))
            result_frame[key] =   [MRR_dictionary[key].score()]
            
            
        # get the results of MRR values.....    
        for key in HR_dictionary:
            print(key +"   "+ str(  HR_dictionary[key].score()    ))
            result_frame[key] = [HR_dictionary[key].score()]
        name = "TAGNN_SR_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path / name, sep = "\t", index = False) 
        
       
        
        
        
        
        


