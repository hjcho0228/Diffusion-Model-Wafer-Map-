from sklearn.model_selection import train_test_split
import os
from tqdm.auto import trange, tqdm
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import KFold
plt.style.use('seaborn-v0_8')
sns.set(font_scale=2) 


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from PIL import Image

# Data
df = pd.read_pickle("../datasets/LSWMD.pkl")

# Drop column
df = df.drop(['waferIndex'], axis = 1)

# 2. Add wafermapDim column because waferMap dim is different each other.
def find_dim(x):
    dim0 = np.size(x, axis=0)
    dim1 = np.size(x, axis=1)
    return dim0, dim1
df['waferMapDim']= df['waferMap'].apply(lambda x: find_dim(x))

# 3. To check failureType distribution and encoding label
df['failureNum'] = df['failureType']
mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
df = df.replace({'failureNum':mapping_type})

# df_withlabel : 결함이 있는 웨이퍼만 선택
# df_withpattern : labeled & patterned wafer : 결함에서 특정 패턴이 나타나는 웨이퍼만 선택
# df_nonpatter : labeled but non-patterned wafer : 결함에서 아무런 패턴도 나타나지 않는 웨이퍼만 선택
df_withlabel = df[(df['failureType']!=0)]
df_withlabel =df_withlabel.reset_index() #labeled index.
df_withpattern = df_withlabel[(df_withlabel['failureType'] != 'none')]
df_withpattern = df_withpattern.reset_index() #patterned index.
df_nonpattern = df_withlabel[(df_withlabel['failureType'] == 'none')] #nonpatterned index
df_withlabel.shape[0], df_withpattern.shape[0], df_nonpattern.shape[0] # 위 각 웨이퍼 수 : A = B + C


# Extract (25,27) & (26,26) waferMapDim data
def subwafer(sw,label):
    Dim0 = np.size(sw, axis=1)
    Dim1 = np.size(sw, axis=2)
    sub_df = df_withlabel.loc[df_withlabel['waferMapDim'] == (Dim0, Dim1)] #sub_df에는 waferMapDim이 (25,27) 또는 (26,26)인 행만 저장됨
    sub_wafer = sub_df['waferMap'].values 
    sw = sw.to(torch.device('cuda'))
    for i in range(len(sub_df)):
        waferMap = torch.from_numpy(sub_df.iloc[i,:]['waferMap'].reshape(1, Dim0, Dim1)) # 이 시점에서 패턴이 없는 웨이퍼는 필터링되면서 제외됨
        waferMap = waferMap.to(torch.device('cuda'))
        sw = torch.cat([sw, waferMap])
        label.append(sub_df.iloc[i,:]['failureType'][0][0])
    x = sw[1:]
    y = np.array(label).reshape((-1,1))
    del waferMap, sw
    return x, y # x: 웨이퍼 맵 데이터, y: 결함 유형(numpy 배열)

sw0 = torch.ones((1, 25, 27))
sw1 = torch.ones((1, 26, 26))
label0 = list()
label1 = list()
x0, y0 = subwafer(sw0, label0)
x1, y1 = subwafer(sw1, label1)
print("x0.shape:", x0.shape)
print("y0.shape:", y0.shape)
print("x1.shape:", x1.shape)
print("y1.shape:", y1.shape)


# Add RGB space for one-hot encoding
# 0: non wafer -> R, 1: normal die -> G, 2: defect die -> B
def rgb_sw(x):
    Dim0 = np.size(x, axis=1)
    Dim1 = np.size(x, axis=2)
    new_x = np.zeros((len(x), Dim0, Dim1, 3))
    x = torch.unsqueeze(x,-1)
    x = x.to(torch.device('cpu'))
    x = x.numpy()
    for w in range(len(x)): 
        for i in range(Dim0):
            for j in range(Dim1):
                new_x[w, i, j, int(x[w, i, j].item())] = 1
    return new_x

rgb_x0 = rgb_sw(x0)
rgb_x1 = rgb_sw(x1)

print(f"rgb_x0.shape: {rgb_x0.shape}, rgb_x1.shape: {rgb_x1.shape}")


#To use two dim, we have to resize these data.
def resize(x):
    rwm = torch.ones((1,56,56,3))
    for i in range(len(x)):
        rwm = rwm.to(torch.device('cuda'))
        a = Image.fromarray(x[i].astype('uint8')).resize((56,56))
        a = np.array(a).reshape((1,56,56,3))
        a = torch.from_numpy(a)
        a = a.to(torch.device('cuda'))
        rwm = torch.cat([rwm, a])
    x = rwm[1:]
    del rwm
    return x

resized_x0 = resize(rgb_x0)
resized_x1 = resize(rgb_x1)

print(f"resized_x0.shape: {resized_x0.shape}, resized_x1.shape: {resized_x1.shape}")

# Concatenate all data together
resized_wm = torch.cat([resized_x0, resized_x1])
label_wm = np.concatenate((y0, y1))

print(f"resized_wm.shape: {resized_wm.shape}, label_wm.shape: {label_wm.shape}")

resized_wm_np = resized_wm.cpu().numpy().astype(np.uint8)
label_wm_np = np.vectorize(mapping_type.get)(label_wm).astype(np.uint8).reshape(-1,1)


parent_dir = os.path.dirname(os.getcwd())                   # 현재 디렉토리의 부모 디렉토리 경로 가져오기
datasets_dir = os.path.join(parent_dir, 'datasets')         # 부모 디렉토리에 'datasets' 폴더 추가
output_file = os.path.join(datasets_dir, "WaferMap_forDiffusionTraining.npz")
np.savez(output_file, arr_0=resized_wm_np, arr_1=label_wm_np)

print(f"Save Complete: {output_file}")
