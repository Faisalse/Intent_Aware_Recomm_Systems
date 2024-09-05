# coding:utf-8
from __future__ import absolute_import
import tensorflow as tf
import os
import time
from CSRM_SIGIR2019.csrm import CSRM
import argparse
import CSRM_SIGIR2019.data_process
import numpy as np
from pathlib import Path 
from CSRM_SIGIR2019.rsc15_data_preprocessing import *
import pandas as pd
data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Run CSRM.")
    parser.add_argument('--dataset', nargs='?', default='yoochoose1_64',
                        help='Choose a dataset.')
    parser.add_argument('--model', nargs='?', default='CSRN',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--n_items', type=int, default=39164,
                        help='Item size 37484, 39164')
    parser.add_argument('--dim_proj', type=int, default=150,
                        help='Item embedding dimension. initial:50')
    parser.add_argument('--hidden_units', type=int, default=150,
                        help='Number of GRU hidden units. initial:100')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epoch to wait before early stop if no progress.')
    parser.add_argument('--display_frequency', type=int, default=100,
                        help='Display to stdout the training progress every N updates.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--keep_probability', nargs='?', default='[0.75,0.5]',
                        help='Keep probability (i.e., 1-dropout_ratio). 1: no dropout.')
    parser.add_argument('--no_dropout', nargs='?', default='[1.0,1.0]',
                        help='Keep probability (i.e., 1-dropout_ratio). 1: no dropout.')
    parser.add_argument('--memory_size', type=int, default=512,
                        help='.')
    parser.add_argument('--memory_dim', type=int, default=100,
                        help='.')
    parser.add_argument('--shift_range', type=int, default=1,
                        help='.')
    parser.add_argument('--controller_layer_numbers', type=int, default=0,
                        help='.')
    return parser.parse_args()

def get_validation_data(trining_data, ratio = 0.1):
     train_x, train_y = trining_data[0], trining_data[1]
     test_records = int(len(trining_data[0]) * ratio)
     train_tr = [train_x[ : -test_records], train_y[ : -test_records]   ]
     train_val = [train_x[ -test_records : ], train_y[ -test_records : ]   ]
     return train_tr, train_val
def main():
    # 指定运行的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    args = parse_args()
    load_data_start_time = time.time()
    # 载入数据集
    if args.dataset == 'diginetica':
         pass
    elif args.dataset == 'yoochoose1_64' or args.dataset == 'yoochoose1_4':
        name = "yoochoose-clicks.dat"
        dataset = load_data_rsc15(data_path / name)
        ratio = int(args.dataset.split("_")[1])
        filter_data_ = filter_data_rsc15(dataset, ratio)
        full_train_data_dl, test_data_dl, item_no, train_valid_tr_dl, train_valid_dl, full_train_data, test_data, train_valid, test_valid = split_data_rsc15(filter_data_)
        n_node = item_no
    else:
        n_node = 310
    print("Loading data done. (%0.3f s)" % (time.time() - load_data_start_time))
    print("%d train examples." % len(full_train_data_dl[0]))
    print("%d valid examples." % len(train_valid_dl[0]))
    print("%d test examples." % len(test_data_dl[0]))
    keep_probability = np.array(args.keep_probability)
    no_dropout = np.array(args.no_dropout)
    # Build model
    with tf.Session(config=config) as sess:
        # 建立模型
        model = CSRM(
            sess=sess,
            n_items= n_node,
            dim_proj=args.dim_proj,
            hidden_units=args.hidden_units,
            patience=args.patience,
            memory_size=args.memory_size,
            memory_dim=args.memory_dim,
            shift_range=args.shift_range,
            controller_layer_numbers=args.controller_layer_numbers,
            batch_size=args.batch_size,
            epoch=args.epoch,
            lr=args.lr,
            keep_probability=keep_probability,
            no_dropout=no_dropout,
            display_frequency=args.display_frequency)
        # model traning....
        best_epoch = model.train(train_valid_tr_dl, train_valid_dl, number_epoch = args.epoch, validation = True, stopping_patience = 5)
        accuracy_values  = model.train(full_train_data_dl, test_data_dl, number_epoch = best_epoch)
        result_frame = pd.DataFrame()
        for key in accuracy_values:
            print(key +"   "+ str(  accuracy_values[key].score()    ))
            result_frame[key] = [accuracy_values[key].score()]
        name = args.model+"_"+args.dataset+".txt"
        result_frame.to_csv(result_path / name, sep = "\t", index = False)
        print("end.............")
if __name__ == '__main__':
    main()