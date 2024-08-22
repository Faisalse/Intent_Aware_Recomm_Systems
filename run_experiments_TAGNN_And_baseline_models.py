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
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
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
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    
    model = trans_to_cuda(SessionGraph(opt, n_node))
    model = model_training(model, train_data, opt.epoch)
    
    # intialize MRR class
    MRR_dictionary = dict()
    for i in opt.MRR:
        MRR_dictionary["MRR_"+str(i)] = MRR(i)
    # intialize HR class
    HR_dictionary = dict()
    for i in opt.HR:
        HR_dictionary["HR_"+str(i)] = HR(i)
    print('************* VSTAN: start predicting   ***************')        
    model.eval()
    slices = test_data.generate_batch(1)
    for i in slices:
        target, scores = forward(model, i, test_data)
        sub_scores = scores.topk(200)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        sub_scores = np.ravel(sub_scores)
        target = target[0] -1
        recommendation_list = pd.Series([0 for i in range(len(sub_scores))], index = sub_scores)
        # Calculate the MRR values
        for key in MRR_dictionary:
            MRR_dictionary[key].add(recommendation_list, target)
        # Calculate the HR values
        for key in HR_dictionary:
            HR_dictionary[key].add(recommendation_list, target) 
    # get the results of MRR values.....
    result_frame = pd.DataFrame()    
    for key in MRR_dictionary:
        print(key +"   "+ str(  MRR_dictionary[key].score()    ))
        result_frame[key] =   [MRR_dictionary[key].score()]
    # get the results of MRR values.....    
    for key in HR_dictionary:
        print(key +"   "+ str(  HR_dictionary[key].score()    ))
        result_frame[key] = [HR_dictionary[key].score()]
    name = opt.model+"_"+opt.dataset+".txt"
    result_frame.to_csv(data_path / name, sep = "\t", index = False)
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







