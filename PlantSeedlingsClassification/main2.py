"""
使用特征工程（特征提取算法）进行模型构建

"""


import torch
import torch.nn as nn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import time
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch.optim as optim
from torchvision import transforms, datasets, models
import os
import pickle 

train_database_path= 'E:/py/MachineLearing/MachineLearning-CourseExercise/PlantSeedlingsClassification/train'
test_database_path = 'E:/py/MachineLearing/MachineLearning-CourseExercise/PlantSeedlingsClassification/test'

test_data = []
test_file_name = []
test_database = {}
train_database = {}
train_data = []
train_target = []

# 测试集生成
target_shape =(200,200)# 目标生成的图片大小
for png in os.listdir(test_database_path):
    # 1 == cv2.IMREAD_COLOR
    # 0 == cv2.IMREAD_GRAYSCALE
    # 2 ==cv2.IMREAD_UNCHANGED
    png_data = cv2.imread(os.path.join(test_database_path,png),1)
    resize_data=cv2.resize(png_data,target_shape)
    assert resize_data.shape ==(target_shape[0], target_shape[1],3)
    test_data.append(resize_data)
    test_file_name.append(png)
test_database['data'] = test_data
test_database['file_name']=test_file_name 

# 训练集生成
train_dict=os.listdir(train_database_path)
print(train_dict)
i=0
for dict in train_dict:
    for png in os.listdir(os.path.join(train_database_path,dict)):

        png_data = cv2.imread(os.path.join(train_database_path,dict,png),1)
        resize_data = cv2.resize(png_data,target_shape)
        assert resize_data.shape ==(target_shape[0], target_shape[1],3)
        train_data.append(resize_data)
        train_target.append(i)
    i += 1
train_database['data'] = train_data
train_database['target']=train_target
train_database['dict'] = train_dict
pickle.dump(train_database, open('my_train.pkl','wb'))
pickle.dump(test_database, open('my_test.pkL','wb'))
