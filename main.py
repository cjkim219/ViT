import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transfroms
import subprocess

from dataset import *

from tqdm import tqdm

import config as con
import model as m

from vit_pytorch import ViT
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)
print(device + " is available")
 

# CIFAR10 데이터셋 로드
dataset = CIFAR10_dataset(con.batch_size)
train_loader, _ = dataset.data_loader()

model = m.ViT(con.Hidden_dim_base, con.num_heads, con.batch_size, con.CIFAR10_img_size).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# model.load_state_dict(torch.load(con.model_path))

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = con.learning_rate)
seheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=con.epochs, eta_min=con.eta_min)
 
for epoch in range(con.epochs):
    avg_cost = 0

    pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{con.epochs}", unit="batch")
    
    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)
        hypothesis = model(data)
        optimizer.zero_grad()
        cost = criterion(hypothesis, target)
        cost.backward()
        optimizer.step()
        avg_cost += cost / len(train_loader)
        pbar.set_postfix_str(f"Cost: {cost:.4f}")
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))
    seheduler.step()
 

    if ((epoch + 1)%5 == 0):
        torch.save(model.state_dict(), con.model_path)
        subprocess.run(["python", "test.py"])