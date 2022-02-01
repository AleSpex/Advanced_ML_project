import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch

# Feature Generator
class Feature_base(nn.Module):
    def __init__(self):
        super(Feature_base, self).__init__()

        self.model_resnet = models.resnet18(pretrained=True)
        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# Classifier
class Standard_Predictor(nn.Module):
    def __init__(self,args):
        super(Standard_Predictor, self).__init__()
        self.fc3 = nn.Linear(512,args.num_classes)

    def forward(self, x):
        x = self.fc3(x)
        return x
