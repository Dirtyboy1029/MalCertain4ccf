# -*- coding: utf-8 -*- 
# @Time : 2024/9/19 15:08 
# @Author : DirtyBoy 
# @File : build_dataset_follow_date.py
import pandas as pd
import os
from utils import get_dex_timestamp
from tqdm import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_type', '-dt', type=str, default="malware")
    parser.add_argument('-feature_type', '-ft', type=str, default='drebin')
    args = parser.parse_args()
    data_type = args.data_type
    feature_type = args.feature_type

    malware_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/MalCertain/malware'
    benign_dir = '/home/public/rmt/heren/experiment/cl-exp/LHD_apk/MalCertain/benign'

    malware = os.listdir(malware_dir)
    malware = [os.path.join(malware_dir, item) for item in malware if item.endswith('apk')]

    benign = os.listdir(benign_dir)
    benign = [os.path.join(benign_dir, item) for item in benign if item.endswith('apk')]
    if data_type == 'malware':
        software_list = []
        timestamp_list = []
        for i, item in tqdm(enumerate(malware), total=len(malware), desc='Malware'):
            time_stamp = get_dex_timestamp(item)
            if time_stamp and os.path.isfile(os.path.join('/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool',
                                                          os.path.splitext(os.path.basename(item))[
                                                              0] + '.' + feature_type)):
                software_list.append(os.path.splitext(os.path.basename(item))[0])
                timestamp_list.append(time_stamp)

        pd.DataFrame({'sha256': software_list,
                      'time': timestamp_list}).to_csv('database/malware/malware_' + feature_type + '.csv')
    elif data_type == 'benign':
        software_list = []
        timestamp_list = []
        for i, item in tqdm(enumerate(benign), total=len(benign), desc='Benign'):
            time_stamp = get_dex_timestamp(item)
            if time_stamp and os.path.isfile(os.path.join('/home/public/rmt/heren/experiment/cl-exp/LHD_apk/naive_pool',
                                                          os.path.splitext(os.path.basename(item))[
                                                              0] + '.' + feature_type)):
                software_list.append(os.path.splitext(os.path.basename(item))[0])
                timestamp_list.append(time_stamp)

        pd.DataFrame({'sha256': software_list,
                      'time': timestamp_list}).to_csv('database/benign/benign_' + feature_type + '.csv')
