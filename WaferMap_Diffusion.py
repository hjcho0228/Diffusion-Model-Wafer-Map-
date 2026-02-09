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

############################
# GPU 메모리 동적 할당
############################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 사용 가능:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("❌ GPU를 찾을 수 없음, CPU로 진행")

############################
# 1) 데이터 불러오기
############################

data_file_path = "/home/tako/winter25-intern/eunhyung_4F/4_1_Researcher/datasets/WaferMap_forDiffusionTraining.npz"
data = np.load(data_file_path)

arr_0 = data['arr_0']
arr_1 = data['arr_1']

print("arr_0.shape:", arr_0.shape)  # (N,56,56,3)
print("arr_1.shape:", arr_1.shape)  # (N,1)

############################
# 2) Train/Val/Test Split
############################
# (라벨이 꼭 필요한 건 아니지만 예시로 분할)
X_train, X_test, y_train, y_test = train_test_split(
    arr_0, arr_1, test_size=0.2, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, shuffle=True
)

print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)


parent_dir = os.path.dirname(os.getcwd())
datasets_dir = os.path.join(parent_dir, 'datasets')
os.makedirs(datasets_dir, exist_ok=True)
output_file = os.path.join(datasets_dir, "WaferMap_forDiffusionTest.npz")

# 데이터 저장
np.savez(output_file,
         X_train=X_train,
         X_val=X_val,
         X_test=X_test,
         y_train=y_train,
         y_val=y_val,
         y_test=y_test)

print(f"Data saved successfully at: {output_file}")


############################
# 3) Diffusion 관련 파라미터
############################
IMG_SIZE = 56
BATCH_SIZE = 32
timesteps = 16
time_bar = 1 - np.linspace(0,1.0,timesteps+1)

def forward_noise(x, t):
    """
    x : (batch,56,56,3)
    t : (batch,) in [0..timesteps-1]
    """
    a = time_bar[t]    # a가 숫자가 더 큼
    b = time_bar[t+1]  # b가 숫자가 더 작음

    noise = np.random.randn(*x.shape)  # (batch,56,56,3)

    a = a.reshape((-1,1,1,1))
    b = b.reshape((-1,1,1,1))

    img_a = x*(1-a) + noise*a # 노이즈가 더 많이 껴 있음
    img_b = x*(1-b) + noise*b # 노이즈가 더 적게 껴 있음
    return img_a, img_b

def generate_ts(num):
    return np.random.randint(0, timesteps, size=num)


############################
# 4) 모델 구성
############################
def block(x_img, x_ts):
    """
    x_img : (batch,56,56,filters), noisy input image
    x_ts  : (batch, some_dim), 
    """
    x_parameter = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x_img)
    x_out       = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x_img)

    time_parameter = layers.Dense(128)(x_ts)
    x_parameter    = layers.Multiply()([x_parameter, time_parameter])
    x_out          = layers.Add()([x_out, x_parameter])

    x_out = layers.LayerNormalization()(x_out)
    x_out = layers.Activation('relu')(x_out)
    
    return x_out

def make_model():
    x_input    = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='x_input') # (56,56,3)
    x_ts_input = layers.Input(shape=(1,), name='x_ts_input') #(1,)

    x_ts = layers.Dense(192)(x_ts_input)                    #(192,)
    x_ts = layers.LayerNormalization()(x_ts)                #(192,)
    x_ts = layers.Activation('relu')(x_ts)                  #(192,)

    # ----- down -----
    x = x56 = block(x_input, x_ts)  # (56,56,128)
    x = layers.MaxPool2D(2)(x)      # (28,28,128)

    x = x28 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)      # (14,14,128)

    x = x14 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)      # (7,7,128)

    x = x7 = block(x, x_ts)
    x = layers.MaxPool2D(2)(x)      # (3,3,128)

    x = x3 = block(x, x_ts)

    # MLP middle
    x_flat = layers.Flatten()(x)                   # (None,3*3*128=1152)
    x_cat  = layers.Concatenate()([x_flat, x_ts])  # (1152+192)=1344
    x_cat  = layers.Dense(128)(x_cat)              # (128,)
    x_cat  = layers.LayerNormalization()(x_cat)    # (128,)
    x_cat  = layers.Activation('relu')(x_cat)      # (128,)

    x_cat  = layers.Dense(3*3*32)(x_cat)           # (288,)
    x_cat  = layers.LayerNormalization()(x_cat)    # (288,)
    x_cat  = layers.Activation('relu')(x_cat)      # (288,)
    x_cat  = layers.Reshape((3,3,32))(x_cat)       # (3,3,32)

    # ----- up -----
    x_u = layers.Concatenate()([x_cat, x3])        # (3,3,32) + (3,3,128) = (3,3,160)
    x_u = block(x_u, x_ts)                         # (3,3,128)
    x_u = layers.UpSampling2D(2)(x_u)              # (6,6,128)

    x_u = layers.ZeroPadding2D(((0,1),(0,1)))(x_u) # (7,7,128)
    x_u = layers.Concatenate()([x_u, x7])          # (7,7,256)
    x_u = block(x_u, x_ts)
    x_u = layers.UpSampling2D(2)(x_u)              # (14,14,128)

    x_u = layers.Concatenate()([x_u, x14])         # (14,14,256)
    x_u = block(x_u, x_ts)
    x_u = layers.UpSampling2D(2)(x_u)              # (28,28,128)

    x_u = layers.Concatenate()([x_u, x28])         # (28,28,256)
    x_u = block(x_u, x_ts)
    x_u = layers.UpSampling2D(2)(x_u)              # (56,56,128)

    x_u = layers.Concatenate()([x_u, x56])         # (56,56,256)
    x_u = block(x_u, x_ts)                         # (56,56,128)

    x_out = layers.Conv2D(3, kernel_size=1, padding='same')(x_u)  # (56,56,3)

    model = tf.keras.models.Model([x_input, x_ts_input], x_out)
    return model

############################
# 5) 모델 생성 & 컴파일
############################

with tf.device('/GPU:0'):  # 명시적으로 GPU 사용
    model = make_model()

    def mse_loss(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
    model.compile(loss=mse_loss, optimizer=optimizer)
    


############################
# 6) 학습용 함수
############################
def train_one(x_img):
    """
    x_img: (batch,56,56,3)
    """
    t = generate_ts(BATCH_SIZE)          # (batch_size,)  example: [5,2,8,...]
    x_a, x_b = forward_noise(x_img, t)   # (batch,56,56,3), (batch,56,56,3)
    x_ts = np.array(t).reshape(-1,1)     # (batch_size,1) example: [[5],[2],[8],...]

    # GPU에서 학습
    with tf.device('/GPU:0'):
        loss = model.train_on_batch([x_a, x_ts], x_b)
    return loss

def train(R=50):
    """
    한 에포크를 R번 반복, each epoch => total=50 iteration
    """
    bar = trange(R)
    total = 100
    avg_loss = 0.0

    for i in bar:
        for j in range(total):
            idx = np.random.randint(0, len(X_train), size=BATCH_SIZE)
            x_img = X_train[idx]         # (batch,56,56,3)
            loss = train_one(x_img)      
            avg_loss += loss
            if j%5==0:
                bar.set_description(f"loss: {loss:.5f}")
    
    return avg_loss/(R*total)



############################
# 7) 학습 루프
############################
EPOCHS = 30
loss_history = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    avg_loss = train()

    loss_history.append(avg_loss)

    # learning rate decay
    lr = model.optimizer.learning_rate.numpy()
    new_lr = max(1e-6, lr*0.9)
    model.optimizer.learning_rate.assign(new_lr)

    # 10 에포크마다 모델 저장
    if (epoch+1) % 10 == 0:
        model.save(f"DiffusionModel_epoch_{epoch+1}.h5")
        print(f"Model saved: DiffusionModel_epoch_{epoch+1}.h5")

# Loss 그래프
plt.figure(figsize=(10,5))
plt.plot(range(1,EPOCHS+1), loss_history, marker='o', linestyle='-', color='y', label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()