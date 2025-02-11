import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import KFold
import time
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
from torch_poly_lr_decay import PolynomialLRDecay
import random

# 병렬로 사용할 수
torch.set_num_threads(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# labels 에는 답
labels_df = pd.read_csv('../dacon_data/dirty_mnist_answer.csv')[:]
imgs_dir = np.array(sorted(glob.glob('../dacon_data/dirty/train/*')))[:]
labels = np.array(labels_df.values[:,1:])

test_imgs_dir = np.array(sorted(glob.glob('../dacon_data/test_dirty_mnist_2nd/*')))

imgs=[]
for path in tqdm(imgs_dir[:]):
    img=cv2.imread(path, cv2.IMREAD_COLOR)
    imgs.append(img)
imgs=np.array(imgs)

# 저장소에서 load
class MnistDataset_v1(Dataset):
    def __init__(self, imgs_dir=None, labels=None, transform=None, train=True):
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform
        self.train = train
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get
        img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
        img = self.transform(img)
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img
        
        pass
    


# 메모리에서 load
class MnistDataset_v2(Dataset):
    def __init__(self, imgs=None, labels=None, transform=None, train=True):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.train=train
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs)
    
    def __getitem__(self, idx):
        # 1개 샘플 get1
        img = self.imgs[idx]
        img = self.transform(img)
        
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img

# https://dacon.io/competitions/official/235697/codeshare/2363?page=1&dtype=recent&ptype=pub
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  

# EfficientNet -b0(pretrained)
# MultiLabel output

class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNet_MultiLabel, self).__init__()
        self.network = EfficientNet.from_pretrained('efficientnet-b4', in_channels=in_channels)
        self.output_layer = nn.Linear(1000, 26)

    def forward(self, x):
        x = F.relu(self.network(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

# 해당 코드에서는 1fold만 실행

kf = KFold(n_splits=5, shuffle=True, random_state=42)
folds=[]
for train_idx, valid_idx in kf.split(imgs):
    folds.append((train_idx, valid_idx))

### seed_everything(42)

# 5개의 fold 모두 실행하려면 for문을 5번 돌리면 됩니다.
for fold in range(5):
    model = EfficientNet_MultiLabel(in_channels=3).to(device)
#     model = nn.DataParallel(model)
    train_idx = folds[fold][0]
    valid_idx = folds[fold][1]



    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
        ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        ])


    epochs=50
    batch_size= 15       # 자신의 VRAM에 맞게 조절해야 OOM을 피할 수 있습니다.
    
    
    
    # Data Loader
    train_dataset = MnistDataset_v2(imgs = imgs[train_idx], labels=labels[train_idx], transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = MnistDataset_v2(imgs = imgs[valid_idx], labels = labels[valid_idx], transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)       
    
    
    # optimizer
    # polynomial optimizer를 사용합니다.
    # 
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    decay_steps = (len(train_dataset)//batch_size)*epochs
    scheduler_poly_lr_decay = PolynomialLRDecay(optimizer, max_decay_steps=decay_steps, end_learning_rate=1e-6, power=0.9)

    criterion = torch.nn.BCELoss()
    
    
    epoch_accuracy = []
    valid_accuracy = []
    valid_losses=[]
    valid_best_accuracy=0
    # for epoch in range(epochs):
    #     model.train()
    #     batch_accuracy_list = []
    #     batch_loss_list = []
    #     start=time.time()
    #     for n, (X, y) in enumerate((train_loader)):
    #         X = torch.tensor(X, device=device, dtype=torch.float32)
    #         y = torch.tensor(y, device=device, dtype=torch.float32)
    #         y_hat = model(X)
            
            
    #         optimizer.zero_grad()
    #         loss = criterion(y_hat, y)
    #         loss.backward()
    #         optimizer.step()
    #         scheduler_poly_lr_decay.step()

            
    #         y_hat  = y_hat.cpu().detach().numpy()
    #         y_hat = y_hat>0.5
    #         y = y.cpu().detach().numpy()

    #         batch_accuracy = (y_hat == y).mean()
    #         batch_accuracy_list.append(batch_accuracy)
    #         batch_loss_list.append(loss.item())