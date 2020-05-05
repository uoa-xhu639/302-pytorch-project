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


def trainGoogle(log_interval, model, device, train_loader, optimizer, epoch, loss_over_train, acc_over_train):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
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
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        running_acc += pred.eq(targets.view_as(pred)).sum().item()
        if batch_idx % log_interval == 9:
            avg_loss = running_loss / 10
            acc = running_acc / 10
            loss_over_train.append(avg_loss)
            acc_over_train.append(acc)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            running_loss = 0.0
            running_acc = 0.0
    return loss_over_train, acc_over_train

def train(log_interval, model, device, train_loader, optimizer, epoch, loss_over_train,acc_over_train):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        # matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(output,targets)]
        # acc = matches.count(True)/len(matches)
        loss = criterion(output, targets)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        running_acc += pred.eq(targets.view_as(pred)).sum().item()
        if batch_idx % log_interval == 9:
            avg_loss = running_loss/10
            acc = running_acc / 10
            loss_over_train.append(avg_loss)
            acc_over_train.append(acc)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            running_loss = 0.0
            running_acc = 0.0
    return loss_over_train, acc_over_train



def test(model, device, test_loader,loss_over_test, acc_over_test,use_cuda):
    model.eval()
    test_loss = 0
    running_loss = 0.0
    correct = 0
    running_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    prediction = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
        # for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if use_cuda:
                pred_y = torch.max(output, 1)[1].cuda().data
            else:
                pred_y = torch.max(output, 1)[1].data
            loss = criterion(output, target).item()
            running_loss += loss
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            running_acc += pred.eq(target.view_as(pred)).sum().item()

            # if batch_idx % log_interval == 9:
            avg_loss = running_loss
            avg_acc = running_acc
            loss_over_test.append(avg_loss)
            acc_over_test.append(avg_acc)
            running_acc = 0.0
            running_loss = 0.0

    test_loss /= len(test_loader.dataset)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return loss_over_test, acc_over_test, pred_y



        # if batch_idx % 50 == 0:
        #     test_output = model(inputs)
        #
        #     # !!!!!!!! Change in here !!!!!!!!! #
        #     pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
        #
        #     accuracy = torch.sum(pred_y == targets).type(torch.FloatTensor) / targets.size(0)
        #     print('Epoch: ', batch_idx, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)



def main():
    epoches = 10
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = False
    Google = False
    Conv = True
    Alex = False




    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")


    DOWNLOAD_CIFAR10 = False

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=DOWNLOAD_CIFAR10, transform=transforms.Compose([transforms.ToTensor()]))


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100,
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

    loss_over_train = []
    acc_over_train = []
    loss_over_test = []
    acc_over_test = []
    for epoch in range(1, epoches + 1):

        if Google:
            training_loss, training_acc = trainGoogle(log_interval, model, device, train_loader, optimizer, epoch, loss_over_train, acc_over_train)
        else:
            training_loss, training_acc = train(log_interval, model, device, train_loader, optimizer, epoch, loss_over_train, acc_over_train)



        testing_loss, testing_acc, prediction  = test(model, device, test_loader,loss_over_test, acc_over_test, use_cuda)
        scheduler.step()

    # "1" = 'plane'
    # "2"= 'car'
    # "3" = 'bird'
    # "4" = 'cat'
    # "5" = 'deer'
    # "6" = 'dog'
    # "7" = 'frog'
    # "8" = 'horse'
    # "9" = 'ship'
    # "10" = 'truck'
    from sklearn import metrics
    test_y = test_data.targets
    # print(prediction.tolist(), 'prediction number')
    # print(test_y[:len(prediction)], 'real number')
    # Print the confusion matrix
    print(metrics.confusion_matrix(test_y[:len(prediction)], prediction.tolist()))
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(test_y[:len(prediction)], prediction.tolist(), digits=3))



    # print(prediction)
    plt.figure(1)
    # subplot as 2 x 1, and work on the first one.
    plt.subplot(2, 2, 1)
    plt.plot(training_loss)
    plt.ylabel("Training_loss")
    plt.subplot(2, 2, 2)
    plt.plot(training_acc)
    plt.ylabel("Training acc accruacy")
    plt.subplot(2, 2, 3)
    plt.plot(testing_loss)
    plt.ylabel("Testing loss")
    plt.subplot(2, 2, 4)
    plt.plot(testing_acc)
    plt.ylabel("Testing accuracy")
    plt.show()

    # test_y = test_data.test_labels[:2000]
    # print(prediction, 'prediction number')
    # print(test_y[:10].numpy(), 'real number')


    if save_model:
        torch.save(model.state_dict(), "./results/mnist_cnn.pt")


if __name__ == '__main__':
    main()