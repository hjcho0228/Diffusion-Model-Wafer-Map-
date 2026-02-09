from sklearn.model_selection import train_test_split
import os
from tqdm.auto import trange, tqdm
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

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# AutoEncoder Model
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        
        # Encoder
        self.cnn_layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2,2))

        self.cnn_layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2,2))

        # Decoder
        self.tran_cnn_layer1 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding=0),
                                             nn.ReLU())

        self.tran_cnn_layer2 = nn.Sequential(nn.ConvTranspose2d(16, 3, kernel_size = 2, stride = 2, padding=0),
                                             nn.Sigmoid())
            
    def encoder(self, x):
        encode = self.cnn_layer1(x)
        encode = self.cnn_layer2(encode)   
        return encode
    
    def decoder(self, x):
        decode = self.tran_cnn_layer1(x)
        decode = self.tran_cnn_layer2(decode)
        return decode

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output
    
def gen_data(original_image_tensor, label, target_count=10):
    model.eval()
    augmented_images = []
    label_list = []

    with torch.no_grad():
        for _ in range(target_count):
            # 입력 이미지에 약간의 노이즈를 추가하면 다양성 증가
            noise = torch.randn_like(original_image_tensor) * 0.01  # 약한 Gaussian 노이즈
            noisy_input = original_image_tensor + noise
            noisy_input = torch.clamp(noisy_input, 0.0, 1.0)

            # AutoEncoder를 통과시켜 증강 이미지 생성
            output = model(noisy_input)
            output_np = output.squeeze(0).cpu().numpy()
            augmented_images.append(output_np)
            label_list.append(label)
    
    return augmented_images, label_list

    
################################################################################################################################################################|
######################################################################[모델, 데이터셋 로드]##########################################################################
################################################################################################################################################################|

# 모델 및 데이터셋 로드 부분
model = ConvAutoEncoder().to(torch.device('cpu'))  # 항상 CPU에서 모델을 로드
model.load_state_dict(torch.load("../saved_models/autoencoder_epoch30.pth", map_location=torch.device('cpu')))
model.eval()
print("Model loaded successfully!\n")

# 데이터셋 로드 및 CPU로 이동
parent_dir = os.path.dirname(os.getcwd())
datasets_dir = os.path.join(parent_dir, 'datasets')
output_file = os.path.join(datasets_dir, "Resized_wm_Label_wm.npz")
loaded_data = np.load(output_file)

# Load data
resized_wm = loaded_data['resized_wm']
label_wm = loaded_data['label_wm']

# Convert to tensor, set to float, permute the dimensions, and ensure everything is moved to CPU
resized_wm = torch.tensor(resized_wm).float().to(torch.device('cpu'))
resized_wm = resized_wm.permute(0, 3, 1, 2)  # Change the channel dimension order
label_wm = label_wm  # No need to move label_wm to device since it is a numpy array

print("Dataset loaded successfully!\n")

# 기타 데이터 증강 및 원본 이미지를 저장하는 코드가 여기에 추가됩니다.
# 여기서는 model을 CPU에서만 처리하도록 하는 코드 유지

# 원본 이미지 저장
Label_Original_Image = {}
for f in np.unique(label_wm):
    if f == 'none':  # 'none' 라벨은 제외
        continue
    # 해당 라벨에 해당하는 첫 번째 이미지
    idx = np.where(label_wm == f)[0][0]
    Label_Original_Image[f] = resized_wm[idx].cpu().numpy()  # 이미지를 딕셔너리에 저장

# 데이터 증강 생성
Label_Augmented_Image = {}
for f in np.unique(label_wm):
    if f == 'none':  # 'none' 라벨은 제외
        continue
    idx = np.where(label_wm == f)[0][0]
    original_image = resized_wm[idx].cpu().numpy()
    original_image_tensor = torch.tensor(original_image).float().unsqueeze(0).to(torch.device('cpu'))  # CPU로 로드
    augmented_image, _ = gen_data(original_image_tensor, f, target_count=10)
    Label_Augmented_Image[f] = augmented_image[0].detach().cpu().numpy()  # 텐서에서 detach 후 CPU로 이동

# Plot size 설정
fig, axes = plt.subplots(len(Label_Original_Image), 2, figsize=(12, 3 * len(Label_Original_Image)))

# 각 결함 종류에 대해 원본 이미지와 증강된 이미지 시각화
for idx, (label, original_img) in enumerate(Label_Original_Image.items()):
    # 원본 이미지는 첫 번째 열에
    axes[idx, 0].imshow(np.transpose(original_img, (1, 2, 0)))  # 채널을 맞춰서 이미지를 출력
    axes[idx, 0].set_title(f"Original: {label}")
    axes[idx, 0].axis('off')  # 축 숨기기

    # 증강된 이미지는 두 번째 열에
    augmented_img = Label_Augmented_Image[label]  # 증강된 이미지
    axes[idx, 1].imshow(np.transpose(augmented_img, (1, 2, 0)))  # 채널을 맞춰서 이미지를 출력
    axes[idx, 1].set_title(f"Augmented: {label}")
    axes[idx, 1].axis('off')  # 축 숨기기

# 레이아웃 조정
plt.tight_layout()
plt.show()
