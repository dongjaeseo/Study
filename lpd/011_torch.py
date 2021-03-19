import os
import numpy as np
import pandas as pd
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
#딥러닝 모델 설계할 때 장비 확인
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using pytorch version:',torch.__version__,'Device:',DEVICE) #Using pytorch version: 1.7.1 Device: cuda

##cache 비워주기###
import torch,gc
gc.collect()
torch.cuda.empty_cache()

BATCH_SIZE = 32
EPOCHS = 10
TRAIN_PATH = 'C:/data/LPD_competition/train/'

transformer = transforms.Compose([
        transforms.Resize((150,150)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
                            [0.5,0.5,0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

image_datasets = datasets.ImageFolder(TRAIN_PATH, transformer)
#print(image_datasets)

class_names = image_datasets.classes
#print(class_names)

train_size = int(0.8*len(image_datasets))
test_size = len(image_datasets) - train_size

print(train_size) #38400
print(test_size) #9600

train_dataset, test_dataset = torch.utils.data.random_split(image_datasets, [train_size,test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True, )
valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64, shuffle = False, )

root = pathlib.Path(TRAIN_PATH)
classe = sorted([ j.name.split('/')[-1] for j in root.iterdir()])
print(classe)

inputs, classes = next(iter(train_loader))
print(classes)
print(inputs)

# 임의의 label값과 사진 불러오기 
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.5,0.5,0.5])
    std = np.array([0.5,0.5,0.5])
    inp = std*inp +mean
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

out = torchvision.utils.make_grid(inputs)
#imshow(out, title=[class_names[x] for x in classes])
#plt.show()

class ConvNet(nn.Module):
    def __init__(self , num_classes = 1000):
        super(ConvNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3 , out_channels=12 , kernel_size=3, stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=12 , out_channels=20 , kernel_size=3, stride=1,padding=1)
        self.relu2 = nn.ReLU()
         
        self.conv3 = nn.Conv2d(in_channels=20 , out_channels=32 , kernel_size=3, stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        
        self.fc = nn.Linear(in_features= 32*75*75, out_features=num_classes)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
            
        output = self.pool(output)
            
        output = self.conv2(output)
        output = self.relu2(output)
            
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
            
        output = output.view(-1 , 32*75*75)
        output = self.fc(output)
            
        return output

from efficientnet_pytorch import EfficientNet
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet =  EfficientNet.from_pretrained('efficientnet-b3')
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256,1000)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= Classifier().to(device)
#model = ConvNet(num_classes = 1000).to(device)
optimizer = Adam(model.parameters(),lr=0.001 , weight_decay=0.0001)
loss_function = nn.CrossEntropyLoss()

num_epochs=20

best_accuracy=0.0
'''
for epoch in tqdm(range(num_epochs)):
    
    #Evaluation and training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/train_size
    train_loss=train_loss/train_size
    
    
    # Evaluation on testing dataset
    model.eval()
    
    test_accuracy=0.0
    for i, (images,labels) in enumerate(valid_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))
    
    test_accuracy=test_accuracy/test_size
    
    
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy)+' Test Accuracy: '+str(test_accuracy))
    
    #가장 좋은 모델 저장
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'best_checkpoint.pth')
        best_accuracy=test_accuracy
'''
# import natsort as nt   #.txt 파일을 순서대로 정렬
# from PIL import Image
# from torch.autograd import Variable
# def test_model():
#     data_transforms = transforms.Compose([
#         transforms.Resize((150,150)),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5,0.5,0.5],
#                             [0.5,0.5,0.5])
#     ])
#     model_ft = torch.load('C:/data/LPD_competition/pt/best_checkpoint.pt', map_location=device)
PATH = 'C:/data/LPD_competition/pt/best_checkpoint.pt'
#model = torch.load('C:/data/LPD_competition/pt/best_checkpoint.pt', map_location=device)
model.load_state_dict(torch.load(PATH),strict=False)
#RuntimeError: Error(s) in loading state_dict for Classifier: 오류
#pretrained pytorch model을 loading해오려고 했는데, pytorch version과 여러 환경세팅이 맞지 않아서 모델의 state_dict에 있는 key가 matching이 되지 않아 모델의 pretrained weight가 불려오지 않는 문제


correct = 0
total = 0

with torch.no_grad():

    for data in valid_loader:

        images, labels = data[0].to(device), data[1].to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        print(predicted)
        total += labels.size(0)
        
        
        correct += (predicted == labels).sum().item()
        for i, j in enumerate(predicted):
            print(i)
            print(j)
            print(labels[i])
        
                
        print(total)
        print(correct)

print("Accuracy of the network on th 10000 test images : %d %%" % (100 * correct / total))

#output = test_model()
submission = pd.read_csv('C:/data/LPD_competition/csv/sample.csv')
#submission.digit = torch.cat(output).detach().cpu().numpy()
submission.to_csv('C:/data/LPD_competition/submission/pytorch_result.csv', index=False)