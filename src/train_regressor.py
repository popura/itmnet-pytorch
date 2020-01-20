import time
import argparse
import functools
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from deepy.pytorch.model import ITMNet
from deepy.pytorch.trainer import RegressorTrainer
from deepy.pytorch.dataset import CaiMEImageDataset
import deepy.pytorch.layer as layer

parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', help='Output directory of models.',
                    default="../model")
parser.add_argument('--data', help='Data directory.',
                    default="../data")
parser.add_argument('--batch_size', help='Batch size',
                    type=int,
                    default=64)
parser.add_argument('--model', help='Model name for predicting images.',
                    default="ITMNet")
parser.add_argument('--dn_layer', help='Downsampling layer name.',
                    default="Conv2d")
parser.add_argument('--dn_hold_mode', help='Location of Hold layer.',
                    default="hold_first")
parser.add_argument('--dn_bias_mode', help='Lacation of bias.',
                    default="bias_first")
parser.add_argument('--dn_hold_order', help='Order of hold kernel.',
                    type=int,
                    default=0)
parser.add_argument('--loss',
                    help='Loss function for training',
                    default="cross_entropy")
parser.add_argument('--seed', help='Random seed',
                    type=int,
                    default=None)
parser.add_argument('--epoch', help='Number of epochs',
                    type=int,
                    default=300)
parser.add_argument('--optimizer', help='Optimizer',
                    default='Adam')
parser.add_argument('--lr', help='Initial learning rate',
                    type=float,
                    default=0.1)
args = parser.parse_args()
output_dir = args.o
data_dir = args.data
model_name = args.model
dn_layer = args.dn_layer
if dn_layer == 'Conv2d':
    dn_hold_mode = None
    dn_bias_mode = None
    dn_order = None
else:
    dn_hold_mode = args.dn_hold_mode
    dn_bias_mode = args.dn_bias_mode
    dn_order = args.dn_hold_order
batch_size = args.batch_size
loss_name = args.loss
random_seed = args.seed
num_epochs = args.epoch
optimizer_name = args.optimizer
learning_rate = args.lr

print('-----Parameters-----')
print('model name: ' + model_name)
print('down conv layer: ' + dn_layer)
print('hold mode: ' + str(dn_hold_mode))
print('hold order: ' + str(dn_order))
print('bias mode: ' + str(dn_bias_mode))
print('batch size: %d' % batch_size)
print('loss: ' + loss_name)
print('seed: %d' % random_seed)
print('optimizer: ' + optimizer_name)
print('learning rate: %f' % learning_rate)

if random_seed is not None:
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = CaiMEImageDataset(root=data_dir, train=True,
                             transform=transform, target_transform=target_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if dn_layer == "Conv2d":
    down_conv_layer = nn.Conv2d
elif dn_layer == "DownSampling2d":
    down_conv_layer = functools.partial(layer.DownSampling2d,
                                        order=dn_order,
                                        hold_mode=dn_hold_mode,
                                        bias_mode=dn_bias_mode)
else:
    raise(NotImplementedError())

if model_name == "MyNet":
    net = model.MyNet(down_conv_layer=down_conv_layer)
elif "VGG" in model_name:
    net = model.VGG(model_name, down_sampling_layer=down_conv_layer)
elif "ResNet" in model_name:
    net = model.ResNet(model_name, down_sampling_layer=down_conv_layer)
else:
    raise(NotImplementedError())

net = net.to(device)
summary(net, input_size=(3, 32, 32))

if loss_name == "cross_entropy":
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.CrossEntropyLoss()

if optimizer_name == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
elif optimizer_name == "SGD":
    optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                          momentum=0.9, dampening=0,
                          weight_decay=0.0001, nesterov=False)
else:
    raise(NotImplementedError())

scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           [num_epochs//2, 3*num_epochs//4],
                                           0.1)
start_time = time.time()

trainer = trainer.ClassifierTrainer(net, optimizer, criterion,
                                    trainloader, device)
costs = []
train_accuracy = []
test_accuracy = []
print('-----Training Started-----')
for epoch in range(num_epochs):  # loop over the dataset multiple times
    loss = trainer.train()
    train_acc = trainer.eval(trainloader)
    test_acc = trainer.eval(testloader)
#    if epoch == 4 and model_name == "Net":
#        trainer.visualize_grad(trainloader)
    print('Epoch: %03d/%03d | Loss: %.4f | Time: %.2f min | Acc: %.4f/%.4f'
          % (epoch+1, num_epochs, loss,
             (time.time() - start_time)/60,
             train_acc, test_acc))
    costs.append(loss)
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)
    scheduler.step()

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

if dn_layer == 'Conv2d':
    output_model_name = (model_name + '_' + dn_layer
                         + '_' + optimizer_name + '.pth')
else:
    output_model_name = (model_name + '_' + dn_layer + '_' + dn_hold_mode
                         + '_{:d}-order'.format(dn_order)
                         + '_' + dn_bias_mode
                         + '_' + optimizer_name + '.pth')
torch.save(net.state_dict(), output_dir+"/"+output_model_name)

plt.plot(range(len(costs)), costs, label='loss')
plt.legend()
plt.show()

plt.plot(range(len(train_accuracy)), train_accuracy,
         label='Accuracy for training data')
plt.legend()
plt.show()

plt.plot(range(len(test_accuracy)), test_accuracy,
         label='Accuracy for test data')
plt.legend()
plt.show()
