import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from initializers import gaussian_initializer, constant_initializer
import random



class ModelAVE(nn.Module):
    def __init__(self):
        super(ModelAVE, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=9, stride=8)

    def forward(self, center_map):
        x = self.avg_pool(center_map)
        return x



class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, 15, kernel_size=1)
        # self.pool_center_lower = None
        # center_map_np = np.random.randint(2, size=(10, 15, 368, 368))
        # center_map = center_map_np.tolist()
        # center_map_tensor = torch.tensor(center_map, dtype=torch.float16)
        # self.center_map = torch.nn.Parameter(center_map_tensor)
        
    # def _initialize_weights(self):
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 gaussian_initializer()(m.weight)
    #                 if m.bias is not None:
    #                     constant_initializer()(m.bias)

    def forward(self, image):
        x1 = F.relu(self.conv1_stage1(image))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        x1 = F.relu(self.conv2_stage1(x1))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        x1 = F.relu(self.conv3_stage1(x1))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        x1 = F.relu(self.conv4_stage1(x1))
        x1 = F.relu(self.conv5_stage1(x1))
        x1 = F.relu(self.conv6_stage1(x1))
        x1 = self.conv7_stage1(x1)
        return x1
    



class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        # center_map_np = np.random.randint(2, size=(15, 368, 368))
        # center_map = center_map_np.tolist()
        # center_map_tensor = torch.tensor(center_map, dtype=torch.float16)
        # self.center_map = torch.nn.Parameter(center_map_tensor)

    # def _initialize_weights(self):
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv2d):
    #                 gaussian_initializer()(m.weight)
    #                 if m.bias is not None:
    #                     constant_initializer()(m.bias)

    def forward(self, image):
        x2 = F.relu(self.conv1_stage2(image))
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2)
        x2 = F.relu(self.conv2_stage2(x2))
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2)
        x2 = F.relu(self.conv3_stage2(x2))
        x3 = F.max_pool2d(x2, kernel_size=3, stride=2)
        x2 = F.relu(self.conv4_stage2(x3))
        return x2, x3
    

class ModelM2(nn.Module):
    def __init__(self):
        super(ModelM2, self).__init__()
        self.Mconv1 = nn.Conv2d(62, 128, kernel_size=11, padding=5)
        self.Mconv2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4 = nn.Conv2d(128, 128, kernel_size=1)
        self.Mconv5 = nn.Conv2d(128, 15, kernel_size=1)
        center_map_np = np.random.randint(2, size=(15, 368, 368))
        center_map = center_map_np.tolist()
        center_map_tensor = torch.tensor(center_map, dtype=torch.float16)
        self.center_map = torch.nn.Parameter(center_map_tensor)

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)

    def forward(self, image):
        x2 = F.relu(self.Mconv1(image))
        x2 = F.relu(self.Mconv2(x2))
        x2 = F.relu(self.Mconv3(x2))
        x2 = F.relu(self.Mconv4(x2))
        x2 = F.relu(self.Mconv5(x2))
        return x2
    
    
class ModelM3(nn.Module):
    def __init__(self):
        super(ModelM3, self).__init__()
        self.Mconv1 = nn.Conv2d(62, 128, kernel_size=11, padding=5)
        self.Mconv2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4 = nn.Conv2d(128, 128, kernel_size=1)
        self.Mconv5 = nn.Conv2d(128, 15, kernel_size=1)
        center_map_np = np.random.randint(2, size=(15, 368, 368))
        center_map = center_map_np.tolist()
        center_map_tensor = torch.tensor(center_map, dtype=torch.float16)
        self.center_map = torch.nn.Parameter(center_map_tensor)

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)

    def forward(self, image):
        x2 = F.relu(self.Mconv1(image))
        x2 = F.relu(self.Mconv2(x2))
        x2 = F.relu(self.Mconv3(x2))
        x2 = F.relu(self.Mconv4(x2))
        x2 = F.relu(self.Mconv5(x2))
        return x2

    
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.conv1_stage3 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)

    def forward(self, image):
        x3 = F.relu(self.conv1_stage3(image))
        return x3



class Model(nn.Module):
    def __init__(self, model1, model2, model3, modelAVE, model2_M, model3_M):
        super(Model, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.modelAVE = modelAVE
        self.model2_M = model2_M
        self.model3_M = model3_M

        center_map_np = np.random.randint(2, size=(10, 15, 368, 368))
        center_map = center_map_np.tolist()
        center_map_tensor = torch.tensor(center_map, dtype=torch.float16)
        self.center_map = torch.nn.Parameter(center_map_tensor) 
        

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)
    
    def forward(self, x): #x-image
        print('before model1', torch.cuda.memory_reserved())
        output1 = self.model1(x)
        print('after model1', torch.cuda.memory_reserved())
        #print(output1.shape)
        output2, input5 = self.model2(x)
        print('after model2', torch.cuda.memory_reserved())
        #print(output2.shape)
        outputAVE = self.modelAVE(self.center_map)
        print('after modelAVE', torch.cuda.memory_reserved())
        #print(outputAVE.shape)
        concatenated_output = torch.cat([output1, output2, outputAVE], dim=1)
        #print(concatenated_output.shape)
        output3 = self.model2_M(concatenated_output)
        print('after model2_M', torch.cuda.memory_reserved())
        output4 = self.model3(input5)
        print('after model3', torch.cuda.memory_reserved())
        concatenated_output1 = torch.cat([output3, outputAVE, output4], dim=1)
        output5 = self.model3_M(concatenated_output1)
        print('after model3_M', torch.cuda.memory_reserved())
        return output5
        