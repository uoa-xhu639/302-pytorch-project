import torch.nn as nn
import torch.nn.functional as F
import torch

class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, 3)  #3 32 32
        self.conv2 = nn.Conv2d(48, 128, 3) #48 32 32
        self.conv3 = nn.Conv2d(128, 192, 3) #
        self.conv4 = nn.Conv2d(192, 128, 3) #
        # self.conv5 = nn.Conv2d(192, 128, 3) #
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128*4*4, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 10)



    def forward(self, x):
        x = self.conv1(x) #3 32 32
        x = F.relu(x) #48 30 30
        x = F.max_pool2d(x, 2) # 48 15 15
        x = self.conv2(x) #128 13 13
        x = F.relu(x) #
        x = self.conv3(x) #192 11 11
        x = F.relu(x) #
        x = self.conv4(x) #128 9 9
        x = F.relu(x) #
        x = F.max_pool2d(x, 2) #128 4 4
        x = self.dropout(x)
        x = torch.flatten(x, 1) #
        x = self.fc1(x) #
        x = F.relu(x) #
        x = self.dropout(x) #
        x = self.fc2(x) #
        output = F.log_softmax(x, dim=1)
        return output

