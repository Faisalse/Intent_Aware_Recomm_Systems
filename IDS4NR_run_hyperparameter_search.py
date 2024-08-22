#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_import_list import *

import traceback
import os, multiprocessing
from functools import partial

from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.Movielens.Movielens100MReaderGiven import Movielens100MReaderGiven
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid


def run_experiments_for_IDSNR_Model():
    """
    This function provides a simple example on how to tune parameters of a given algorithm
    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

    commonFolderName = "experiments_results"
    model = "IDSNR"
    dataset_name = "Music" # MovieLens, Music, Beauty
    dataSetType = "given"
    task = "optimization"

    validation_set = True
    dataset_object = Movielens100MReaderGiven()
    URM_train, URM_validation, URM_test, UCM_all = dataset_object._load_data_from_give_files(validation=validation_set, data_name = dataset_name)
    saved_results = "/".join([commonFolderName,model,dataset_name, dataSetType, task] )
    # If directory does not exist, create
    if not os.path.exists(saved_results):
        os.makedirs(saved_results+"/")

    # model to optimize
    collaborative_algorithm_list = [
    
        P3alphaRecommender,
        RP3betaRecommender,
        ItemKNNCFRecommender,
        UserKNNCFRecommender

    ]
    from Evaluation.Evaluator import EvaluatorHoldout
    cutoff_list = [5, 10, 20]
    metric_to_optimize = "RECALL"
    cutoff_to_optimize = 10

    n_cases = 35
    n_random_starts = 5

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)

    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       metric_to_optimize = metric_to_optimize,
                                                       cutoff_to_optimize = cutoff_to_optimize,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = saved_results,
                                                       resume_from_saved = True,
                                                       similarity_type_list = ["cosine"],
                                                       parallelizeKNN = False)


    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)



if __name__ == '__main__':
    run_experiments_for_IDSNR_Model()
