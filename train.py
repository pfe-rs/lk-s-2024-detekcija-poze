from utilss import get_accuracy_train, get_accuracy_test, plot_curve, keep_store_dict, store_dict_to_disk
from tqdm import tqdm
from test import test
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
from data_loader import DataLoader1
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def train(model, num_epochs, train_loader, store_dict, test_loader, device, loss_function, optimizer, batch_size):
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_prec = 0.0
        train_rec = 0.0
        model = model.train()
        train_data_loader = DataLoader(train_loader, batch_size=batch_size)#, shuffle=True)
        valid_data_loader = DataLoader(test_loader, batch_size=batch_size) #, shuffle=True)
        #train_loader.used = []
        #for batch_num, (x, y) in tqdm(enumerate(train_data_loader)):
        for batch_num, (x, y, z) in enumerate(train_data_loader):

            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            optimizer.zero_grad() 
            
            y_hat = model(x, z.float())
            m = nn.Sigmoid()
            y_hat = m(y_hat).float()
            y = y.float()
            #print(x.shape, y_hat.shape, y.shape)
            if num_epochs == 55: 
                input1 = y_hat[0]
                input1a = input1.cpu()
                input1 = input1.to(device)
                a = ['r ankle','r knee','r hip', 'l hipl', 'r knee','l ankle','pelvis','thorax','upper neck','head top','r wrist','r elbow', 'r shoulder', 'l shoulder','l elbow','l wrist', 'back']
                for i in range(15):
                    output = input1a[i]
                    output = output.cpu()
                    numpy_array = output.detach().numpy()
                    plt.imshow(numpy_array*255)
                    plt.xlabel(a[i])
                    plt.show()

            # print(y_hat.min(), y_hat.max())

            # if y_hat is None:
            #     raise ValueError("Model output y_hat is None")
            
            
            loss = loss_function(y_hat, y)
    
            train_running_loss += loss
            prec, rec = get_accuracy_train(y_hat, y)
            train_prec += prec
            train_rec += rec
            
            #print(get_accuracy_train(y_hat, y))

            
            loss.backward()
            optimizer.step()
            
            # if batch_num == 0:
            #     break
            
        # for param in model.parameters():
        #     param1 = param.cpu()
        #     store_dict = keep_store_dict(curve=param1, label='after_optimizer_step', store_dict=store_dict)  
            
            
        epoch_loss = train_running_loss / (batch_num + 1)
        epoch_prec = train_prec / (batch_num + 1)
        epoch_rec = train_rec / (batch_num + 1) 
        
        #store_dict = keep_store_dict(curve=epoch_loss, label='train_loss', store_dict=store_dict)
        #store_dict = keep_store_dict(curve=epoch_acc, label='train_acc', store_dict=store_dict)
        print('Epoch: %d | Loss: %.4f | Prec: %.4f | Rec: %.4f' \
              %(epoch + 1, epoch_loss, epoch_prec, epoch_rec))
        v = time.asctime().split()
        v = '_'.join(v)
        torch.save(model,'/notebooks/lk-s-2024-detekcija-poze/models/model_'+ v +'.pth')

        #         if test_loader is not None:
        #             test_acc = test(model=model, test_loader=test_loader, device=device)
        #             store_dict = keep_store_dict(curve=test_acc, label='test_acc', store_dict=store_dict)
        
    return store_dict