import argparse
import pickle
import time
# import files for dl
from TAGNN.TAGNN_code.utils import build_graph, Data, split_validation
from TAGNN.TAGNN_code.model import *
# import files for preprocessing
from TAGNN.data_preprocessing.digi_data_preprocessing import *
from TAGNN.data_preprocessing.rsc15_data_preprocessing import *
# baseline models
from TAGNN.baselines.SR.main_sr import *
# vstan model
from TAGNN.baselines.vstan.main_vstan import *
#stan model....
from TAGNN.baselines.stan.main_stan import *
# sfcknn model
from TAGNN.baselines.sfcknn.main_sfcknn import *
from pathlib import Path
# context free method
from TAGNN.baselines.CT.main_ct import *
# import accuracy measures
from TAGNN.accuracy_measures import *
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='TAGNN')
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--MRR', type=float, default=[5, 10, 20], help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--HR', type=float, default=[5, 10, 20], help='learning rate')  # [0.001, 0.0005, 0.0001]

opt = parser.parse_args()
data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()

def get_validation_data(trining_data, ratio = 0.1):
     train_x, train_y = trining_data[0], trining_data[1]
     test_records = int(len(trining_data[0]) * ratio)
     train_tr = [train_x[ : -test_records], train_y[ : -test_records]   ]
     train_val = [train_x[ -test_records : ], train_y[ -test_records : ]   ]
     return train_tr, train_val
def run_experiments_for_TAGNN():
    if opt.dataset == 'diginetica':
        name = "train-item-views.csv"
        dataset = load_data(data_path / name) 
        filter_data_ = filter_data(dataset)
        train_data, test_data, item_no = split_data(filter_data_)
        n_node = item_no
        
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        name = "yoochoose-clicks.dat"
        dataset = load_data_rsc15(data_path / name)
        filter_data_ = filter_data_rsc15(dataset)
        train_data, test_data, item_no = split_data_rsc15(filter_data_)
        n_node = item_no       
    else:
        n_node = 310

    train_tr, train_val = get_validation_data(train_data)
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    train_tr = Data(train_tr, shuffle=True)
    train_val = Data(train_val, shuffle=False)

    model = trans_to_cuda(SessionGraph(opt, n_node))
    best_epoch = model_training(model, train_data, train_val, opt.epoch, opt.MRR, validation=True)
    # training on whole data
    accuracy_values = model_training(model, test_data, train_val, best_epoch, opt.MRR, validation=False)
    
    # get the results of MRR values.....
    result_frame = pd.DataFrame()
    for key in accuracy_values:
        print(key +"   "+ str(  accuracy_values[key].score()    ))
        result_frame[key] = [accuracy_values[key].score()]
    name = opt.model+"_"+opt.dataset+".txt"
    result_frame.to_csv(result_path / name, sep = "\t", index = False)
if __name__ == '__main__':
    print("Experiments are runing for each model. After execution, the results will be saved into *results*. Thanks for patience.")
    print("Experiments are runinig for TAGNN model................... wait for results...............")
    run_experiments_for_TAGNN()
    print("Experiments are runinig for SR model................... wait for results...............")
    se_obj = SequentialRulesMain(data_path, result_path, dataset = opt.dataset)
    se_obj.fit_(opt.MRR, opt.HR)
    
    print("Experiments are runinig for VSTAN model................... wait for results...............")
    vstan_obj = VSTAN_MAIN(data_path, result_path, dataset = opt.dataset)
    vstan_obj.fit_(opt.MRR, opt.HR)
    
    print("Experiments are runinig for STAN model................... wait for results...............")
    stan_obj = STAN_MAIN(data_path, result_path, dataset = opt.dataset)
    stan_obj.fit_(opt.MRR, opt.HR)
    
    print("Experiments are runinig for SFCKNN model................... wait for results...............")
    sfcknn_obj = SFCKNN_MAIN(data_path, result_path, dataset = opt.dataset)
    sfcknn_obj.fit_(opt.MRR, opt.HR)







