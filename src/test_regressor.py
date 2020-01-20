import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import imshow
import model


parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_dir', help='Model directory.',
                    default="../model")
parser.add_argument('--data', help='Data directory.',
                    default="../data")
parser.add_argument('--batch_size', help='Batch size',
                    type=int,
                    default=16)
parser.add_argument('--model', help='Model name for predicting images.',
                    default="VGG8")
parser.add_argument('--loss',
                    help='Loss function for training',
                    default="cross_entropy")
args = parser.parse_args()
model_dir = args.model_dir
data_dir = args.data
model_name = args.model
batch_size = args.batch_size
loss_name = args.loss

if model_name == "Net":
    net = model.Net()
elif model_name == "CheckerFreeNet":
    net = model.CheckerFreeNet()
elif model_name == "VGG8":
    net = model.VGG("VGG8", "strided")
elif model_name == "CheckerFreeVGG8":
    net = model.CheckerFreeVGG("VGG8", "strided")
elif model_name == "ResNet18":
    net = model.ResNet("ResNet18")
elif model_name == "CheckerFreeResNet18":
    net = model.CheckerFreeResNet("ResNet18")
else:
    raise(NotImplementedError())

net.load_state_dict(torch.load(model_dir+"/"+model_name+".pth"))
net.eval()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = dataiter.next()
# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
