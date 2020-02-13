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
from PIL import Image
from deepy.pytorch.dataset import CaiMEImageDataset
import deepy.pytorch.trainer as trainer
import deepy.pytorch.model as model
import deepy.pytorch.transform as paired_transforms
import deepy.pytorch.layer as layer

parser = argparse.ArgumentParser(description='')
parser.add_argument('-o', help='Output directory of models.',
                    default="../model")
parser.add_argument('--data', help='Data directory.',
                    default="../data")
parser.add_argument('--batch_size', help='Batch size',
                    type=int,
                    default=16)
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
                    default="l1")
parser.add_argument('--seed', help='Random seed',
                    type=int,
                    default=None)
parser.add_argument('--epoch', help='Number of epochs',
                    type=int,
                    default=1000)
parser.add_argument('--optimizer', help='Optimizer',
                    default='Adam')
parser.add_argument('--lr', help='Initial learning rate',
                    type=float,
                    default=0.002)
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

transforms = paired_transforms.Compose(
    [paired_transforms.PairedRandomHorizontalFlip(p=0.5),
     paired_transforms.PairedRandomResizedCrop(size=(), scale=(0.6, 1.0), ratio=(3./4., 4./3.), interpolation=Image.BICUBIC),
     paired_transforms.ToPairedTransform(transforms.ToTensor())])

trainset = CaiMEImageDataset(root=data_dir, train=True, transforms=transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = CaiMEImageDataset(root=data_dir, train=False, transforms=transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

if dn_layer == "Conv2d":
    down_conv_layer = nn.Conv2d
elif dn_layer == "DownSampling2d":
    down_conv_layer = functools.partial(layer.DownSampling2d,
                                        order=dn_order,
                                        hold_mode=dn_hold_mode,
                                        bias_mode=dn_bias_mode)
else:
    raise(NotImplementedError())

if model_name.lower() == "unet":
    net = model.UNet(down_sampling_layer=down_conv_layer)
elif model_name.lower() == "itmnet": 
    net = model.ITMNet()
else:
    raise(NotImplementedError())

net = net.to(device)
summary(net, input_size=(3, 256, 256))

if loss_name.lower() == "l1":
    criterion = nn.L1Loss()
elif loss_name.lower() == "l2":
    criterion = nn.MSELoss()
else:
    criterion = nn.CrossEntropyLoss()

if optimizer_name == "Adam":
    optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999))
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

trainer = trainer.RegressorTrainer(net, optimizer, criterion,
                                   trainloader, device)
train_costs = []
test_costs = []
print('-----Training Started-----')
for epoch in range(num_epochs):  # loop over the dataset multiple times
    train_loss = trainer.train()
    test_loss = trainer.eval(testloader)
    print('Epoch: %03d/%03d | Time: %.2f min | Loss: %.4f/%.4f'
          % (epoch+1, num_epochs,
             (time.time() - start_time)/60,
             train_loss, test_loss))
    train_costs.append(train_loss)
    test_costs.append(test_loss)
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
