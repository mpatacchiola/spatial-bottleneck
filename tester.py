
#Massimiliano Patacchiola 2018
#This is used to train only additional connections of the Adartss algorithm
#placed on top of a pre-trained resnet34 network.

import numpy as np
import os
import sys
#Net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter
from models.resnet import ResNet, BasicBlock, Bottleneck
# ArgParser
import argparse
from time import gmtime, strftime

def return_cifar10_testing(dataset_path, download = False, mini_batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root=dataset_path, train=False,
                                           download=download, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=8)
    return testloader  
    
def main():
    ##Parser
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('--id', default='', type=str, help='experiment ID')
    parser.add_argument('--arch', default='resnet34', type=str, help='architecture type: resnet18/152')
    parser.add_argument('--batch', default=128, type=int, help='mini-batch size')
    parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
    parser.add_argument('--root', default='./', type=str, help='root path')
    parser.add_argument('--data', default='./', type=str, help='data path')
    args = parser.parse_args()
    DEVICE_ID = args.gpu
    MINI_BATCH_SIZE = args.batch
    ROOT_PATH = args.root
    DATASET_PATH = args.data
    NET_TYPE = args.arch
    ID = args.id
    print("[INFO] ID: " + str(ID))
    print("[INFO] Root path: " + str(ROOT_PATH))
    print("[INFO] Dataset path: " + str(DATASET_PATH))
    print("[INFO] Mini-batch size: " + str(MINI_BATCH_SIZE))
    #torch.cuda.set_device(DEVICE_ID)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(DEVICE_ID)
    print('[INFO] Is CUDA available: ' + str(torch.cuda.is_available()))
    print('[INFO] TOT available devices: ' + str(torch.cuda.device_count()))
    print('[INFO] Setting device: ' + str(DEVICE_ID))
    #device = torch.device('cuda:'+str(DEVICE_ID) if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print("[INFO] Torch is using device: " + str(torch.cuda.current_device()))
    ##Generate net
    if(NET_TYPE == 'resnet18'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [2,2,2,2])
    elif(NET_TYPE == 'resnet34'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(BasicBlock, [3, 4, 6, 3])
    elif(NET_TYPE == 'resnet50'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(Bottleneck, [3,4,6,3])
    elif(NET_TYPE == 'resnet101'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(Bottleneck, [3,4,23,3])
    elif(NET_TYPE == 'resnet152'):
        from models.resnet import ResNet, BasicBlock, Bottleneck
        net = ResNet(Bottleneck, [3,8,36,3])
    elif(NET_TYPE == 'resnetSB18'):
        from models.resnet import ResNet, SpatialBottleneckBlock
        net = ResNet(SpatialBottleneckBlock, [2,2,2,2])
    elif(NET_TYPE == 'resnetSB34'):
        from models.resnet import ResNet, SpatialBottleneckBlock
        net = ResNet(SpatialBottleneckBlock, [3, 4, 6, 3])
    else:
        raise ValueError('[ERROR] the architecture type ' + str(NET_TYPE) + ' is unknown.') 
    print("[INFO] Architecture: " + str(NET_TYPE))          
    net.to(device)
    if args.resume:
        print('[INFO] Resuming from checkpoint: ' + str(args.resume))
        checkpoint = torch.load(str(args.resume))
        net.load_state_dict(checkpoint['net'])
    else:
        raise ValueError('[ERROR] You must use --resume to load a checkpoint in order to test the model!')        

    #Load the test set
    testloader = return_cifar10_testing(DATASET_PATH, download=False, mini_batch_size=1000)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #Check
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    #Whole dataset accuracy
    correct = 0
    total = 0
    performance = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if(hasattr(net, 'return_performance')):
                performance = net.return_performance()
    print('Accuracy of the network on the 10000 test images: %.5f %%' % (100 * correct / total))
    if(hasattr(net, 'return_performance')):
        print('Performance: ' + str(performance))
    #Per-Class accuracy
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(10):
        print('Accuracy of %5s : %.5f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        
        
if __name__ == "__main__":
    main() 
