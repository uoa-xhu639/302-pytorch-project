import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models.conv import Net
from models.alexnet import Alexnet
from models.VGG import vgg
from models.GoogleNet import GoogLeNet


def trainGoogle(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        # output = model(inputs)
        logits, aux_logits2, aux_logits1 = model(inputs)
        loss0 = criterion(logits, targets)
        loss1 = criterion(aux_logits1, targets)
        loss2 = criterion(aux_logits2, targets)
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        # loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            test_loss += loss
                # F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



        # if batch_idx % 50 == 0:
        #     test_output = model(inputs)
        #
        #     # !!!!!!!! Change in here !!!!!!!!! #
        #     pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
        #
        #     accuracy = torch.sum(pred_y == targets).type(torch.FloatTensor) / targets.size(0)
        #     print('Epoch: ', batch_idx, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)



def main():
    epoches = 30
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = False
    Google = True
    Conv = False
    Alex = False

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")


    DOWNLOAD_CIFAR10 = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=DOWNLOAD_CIFAR10, transform=transforms.Compose([transforms.ToTensor()]))


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
                                               shuffle=True, **kwargs)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=DOWNLOAD_CIFAR10, transform=transforms.Compose([transforms.ToTensor()]))

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000,
                                              shuffle=True, **kwargs)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    if Google:
        print("Model selected: Googlenet")
        model = GoogLeNet(class_num=10, aux_logits=True).to(device)
    elif Conv:
        print("Model selected: conv")
        model = Net().to(device)
    elif Alex:
        print("Model selected: Alexnet")
        model = Alexnet().to(device)
    else:
        print("Model selected: VGG")
        model_name = "VGG13"
        model = vgg(model_name = model_name, class_num = 10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    #after 1 epoach, the learning rate will times gamma to get lower value to increase the accuracy
    # loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, epoches + 1):

        if Google:
            trainGoogle(log_interval, model, device, train_loader, optimizer, epoch)
        else:
            train(log_interval, model, device, train_loader, optimizer, epoch)

        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")


if __name__ == '__main__':
    main()