import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__() #(3, 32, 32)
        # self.conv1 = nn.Sequential(nn.Conv2d(3,6,5),
        #     # in_channels=3, out_channels=6, kernel_size=5, stride=1, padding='valid', ),
        #                            #padding = (kernel_size-1)/2 = (3-1)/2 = 1)
        #                            #(32,32,32)
        #                                    nn.ReLU(), nn.MaxPool2d(2,2)) #(32,16,16)
        # self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5, 1, 1),
        #                            # (32,16,16) -> (64,16,16)
        #                            nn.ReLU(), nn.MaxPool2d(2,2))  #(64,16,16) -> (64,8,8)
        # self.fc1 = nn.Linear(16*5*5 , 120)
        # self.fc2 = nn.Linear(120 , 84)
        # self.fc3 = nn.Linear(84 , 10)

        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        # 1: input channals 32: output channels, 3: kernel size, 1: stride
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)

        #Height&Wedith of picture affected by (H&W-Kernel size)/stride+padding
        #in this case (32-5)/1+1 = 28

        self.pool = nn.AvgPool2d(2,2)
        # x = torch.randn(3,32,32).view(-1,3,32,32)
        # self._to_linear = None
        # self.convs(x)
        # Fully connected layer: input size, output size
        # self.fc1 = nn.Linear(128*4*4, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # def convs(self,x):
    #     x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
    #     x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
    #     x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
    #
    #     if self._to_linear is None:
    #         self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
    #     return x

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = torch.flatten(x, 1)
        # # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # # output = F.log_softmax(x, dim=1)
        # return x

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

#
# X = torch.rand((32,32))
# X = X.view(-1,28*28)
#
# net=Net()
# print(net)
net = Net()