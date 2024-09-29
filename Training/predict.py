# -*- coding: utf-8 -*- 
# @Time : 2024/9/23 13:02 
# @Author : DirtyBoy 
# @File : predict.py
from core.data_preprocessing import data_preprocessing
from core.ensemble.vanilla import Vanilla
from core.ensemble.bayesian_ensemble import BayesianEnsemble
from core.ensemble.deep_ensemble import DeepEnsemble
from core.ensemble.mc_dropout import MCDropout
from core.ensemble.deep_ensemble import WeightedDeepEnsemble
import argparse, os
import numpy as np

Model_Type = {'vanilla': Vanilla,
              'bayesian': BayesianEnsemble,
              'mc_dropout': MCDropout,
              'ensemble': DeepEnsemble,
              }

Architecture_Type = {'drebin': 'dnn',
                     'opcode': 'text_cnn'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_data_type', '-tdt', type=str, default="correct")
    parser.add_argument('-data_type', '-dt', type=str, default="base")
    parser.add_argument('-feature_type', '-ft', type=str, default="drebin")
    args = parser.parse_args()
    feature_type = args.feature_type
    data_type = args.data_type
    test_data_type = args.test_data_type
    architecture_type = Architecture_Type[feature_type]

    test_dataset, gt_labels, input_dim = data_preprocessing(feature_type=feature_type, data_type=test_data_type,
                                                            is_training_set=False, is_finetune=False)
    for model_type in ['vanilla', 'bayesian', 'mc_dropout', 'ensemble']:
        output_path = 'output/' + feature_type
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        model_ensemble = Model_Type[model_type]
        model = model_ensemble(architecture_type=architecture_type,
                               model_directory="../Model/" + feature_type + '/' + data_type + '/')
        prob = model.predict(test_dataset)
        np.save(os.path.join(output_path, data_type + '_' + model_type + '_' + test_data_type), prob)
    ### CUDA_VISIBLE_DEVICES
