# -*- coding: utf-8 -*- 
# @Time : 2024/9/19 15:44 
# @Author : DirtyBoy 
# @File : split_hash_shell.py
import pandas as pd
from Datasets.utils import save_to_txt

df = pd.read_csv('2012_Benign.csv')

# 将'时间戳'列转换为datetime格式
df['dex_date'] = pd.to_datetime(df['dex_date'])

# 指定分割的日期为2012年6月1日
cutoff_date = pd.Timestamp('2012-06-01')

# 按照时间戳将数据分为两组
before_cutoff = df[df['dex_date'] < cutoff_date]
after_cutoff = df[df['dex_date'] >= cutoff_date]

a = before_cutoff['sha256'].tolist()
a = [item.lower() for item in a]
save_to_txt(a, '2012_benign.txt')
a = after_cutoff['sha256'].tolist()
a = [item.lower() for item in a]
save_to_txt(a, '2013_benign.txt')
