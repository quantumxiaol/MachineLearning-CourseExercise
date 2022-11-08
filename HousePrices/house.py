import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, Normalizer, StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
import sklearn_pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,train_test_split
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.special import boxcox1p
import csv
import sys
import scipy
import warnings
warnings.filterwarnings('ignore')

#原始数据检视
df_train = pd.read_csv(r'./data/train.csv')
df_test = pd.read_csv(r'./data/test.csv')
#控制显示的列范围，查看数据的时候，显示所有数据，而且数据表中没有省略号
pd.options.display.max_columns = 10000
pd.options.display.max_rows = 500

df_train.head()
#查看训练集基本信息
df_train.info()
#查看训练集数据的维度                      

