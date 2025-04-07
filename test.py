import torch
import torch.nn as nn
import torch.nn.functional as F

import model as m
import config as con

from dataset import *


from vit_pytorch import ViT


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.manual_seed(777)
# if device == 'cuda':
#     torch.cuda.manual_seed_all(777)
# print(device + " is available")
 
# model = m.ViT(con.Hidden_dim_base, con.num_heads, con.batch_size, con.CIFAR10_img_size).to(device)

model = ViT(
    image_size=con.CIFAR10_img_size,
    patch_size=con.patch_size,
    num_classes=con.num_classes,
    dim=con.Hidden_dim_base,
    depth=con.Encoder_iter,
    heads=con.num_heads,
    mlp_dim=con.Hidden_dim_base * con.MLP_scale,
    dim_head=con.Hidden_dim_base // con.num_heads,
    dropout=con.Dropout_rate
).to(device)

model.load_state_dict(torch.load(con.model_path))

dataset = CIFAR10_dataset(con.batch_size)
train_loader, test_loader = dataset.data_loader()

# test
model.eval()
with torch.no_grad():
    test_correct = 0
    test_total = 0
    train_correct = 0
    train_total = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        preds = torch.max(out.data, 1)[1]
        train_total += len(target)
        train_correct += (preds==target).sum().item()

    print('Train Accuracy: ', 100.*train_correct/train_total, '%')

    
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        out = model(data)
        preds = torch.max(out.data, 1)[1]
        test_total += len(target)
        test_correct += (preds==target).sum().item()
        
    print('Test Accuracy: ', 100.*test_correct/test_total, '%')
    
    
    