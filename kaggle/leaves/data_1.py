'''
Date: 2021-06-10 12:17:51
LastEditors: Liuliang
LastEditTime: 2021-06-10 12:31:17
Description: 
'''
import pandas as pd

ratio = 0.8

labels_dataframe = pd.read_csv('/home/liuliang/leaves/train.csv')

train = labels_dataframe.iloc[0:int(len(labels_dataframe)*ratio)]

class_name = list(set(labels_dataframe['label']))


labels_dataframe['label'].value_counts()

train['label'].value_counts()