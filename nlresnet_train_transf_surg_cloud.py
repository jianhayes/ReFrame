""" ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: xiejianhang@bjtu.edu.cn; jianhang.xie@my.cityu.edu.hk
"""

import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import math
import os
from model.resnet import ResNet18, ResNet50, ResNet101
from model.non_learnable_resnet import NLGroupResNet18, NLGroupResNet50, NLGroupResNet101
import onnx
from utils.onnx_surgery import Surgery

parser = argparse.ArgumentParser(description='Non-Learnable ResNet train & transformation & surgery in server')
parser.add_argument('--seed', default=0, type=int, help='global random seed (training random seed)')
parser.add_argument('--nl-seed', default=1, type=int, help='non-learnable base random seed (init NL conv random seed)')
parser.add_argument('--mono-map', default='eqdiff', type=str, help='base seed monotonic mapping')
parser.add_argument('--init-option', default='kaiming_normal', type=str, help='NL parameters initialization function')
parser.add_argument('--model', default='nlr18', type=str, help='which model')
parser.add_argument('--initw', default=False, type=bool, help='init learnable (trainable) weights ')
parser.add_argument('--nlcvoutch', default=-1, type=int, help='non-learnable conv output channels')
parser.add_argument('--output', default='./onnx_trasm',
                    type=str, help='name of the exchange dir to save onnx model')
parser.add_argument('--groups', default=4, type=int, help='pointwise convolution groups')
parser.add_argument('--shuffle', default=False, type=bool, help='channel shuffle for NL block layer')
parser.add_argument('--nlcv', default='group', type=str, help='NL Conv')
parser.add_argument('--nlf1', default='max', type=str, help='NL filter1')
parser.add_argument('--nlf2', default='mix_avg', type=str, help='NL filter2')

# model training hyperparameters
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-size', default=128, type=int, help='datasets batch_size')
parser.add_argument('--epochs', default=200, type=int, help='number of training epochs')
parser.add_argument('--sche', default='cos', type=str, help='lr scheduler')
parser.add_argument('--data', default='../../Datasets/CIFAR10', type=str, help='datasets path')
args = parser.parse_args()


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # if you are using multi-GPU DataParallel
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # torch version >= 1.7
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


print("PyTorch version is {}".format(torch.__version__))
print("ONNX version is {}".format(onnx.__version__))

# ----------INITIALIZATION----------
print('----------INITIALIZATION----------')
if args.seed == args.nl_seed:
    raise NotImplementedError(f'global seed {args.seed} and nl base {args.nl_seed} seed should not be equal.')

# global random seed
init_seed(args.seed)
print("global seed is {}".format(torch.initial_seed()))

nl_layers_num = 0
if args.model == 'nlr18':
    nl_layers_num = 16
elif args.model == 'nlr50':
    nl_layers_num = 48
elif args.model == 'nlr101':
    nl_layers_num = 99

# base seed monotonic map to seed filters
seed_filters = [0 for _ in range(nl_layers_num)]
if args.mono_map == 'const':  # Not recommend use same generator seed
    seed_filters = [args.nl_seed for _ in range(nl_layers_num)]
elif args.mono_map == 'eqdiff':
    seed_filters = [args.nl_seed + i for i in range(nl_layers_num)]
elif args.mono_map == 'multi10':
    seed_filters = [(args.nl_seed + i) * 10 for i in range(nl_layers_num)]
elif args.mono_map == 'powof2':
    seed_filters = [int(math.pow(2, args.nl_seed + i)) for i in range(nl_layers_num)]
elif args.mono_map == 'sqrt':
    seed_filters = [math.ceil(math.sqrt(args.nl_seed + i)) for i in range(nl_layers_num)]
elif args.mono_map == 'none':
    seed_filters = None  # Not recommend without seed filters
else:
    raise NotImplementedError(f'seed generated function {args.seed_filter} not implemented.')

if args.seed in seed_filters:
    raise NotImplementedError(f'base seed {args.seed} should not in seed filters.')

print(seed_filters)

print('building model via PyTorch format.')
if args.model == 'r18':
    net = ResNet18(imgsz=32)
elif args.model == 'r50':
    net = ResNet50(imgsz=32)
elif args.model == 'r101':
    net = ResNet101(imgsz=32)
elif args.model == 'nlr18':
    net = NLGroupResNet18(nlconv_outchans=args.nlcvoutch, groups=args.groups, shuffle=args.shuffle,
                          nlcv=args.nlcv, nlf=(args.nlf1, args.nlf2), init_weights=args.initw, imgsz=32,
                          seed_filters=seed_filters, init_option=args.init_option)
elif args.model == 'nlr50':
    net = NLGroupResNet50(nlconv_outchans=args.nlcvoutch, groups=args.groups, shuffle=args.shuffle,
                          nlcv=args.nlcv, nlf=(args.nlf1, args.nlf2), init_weights=args.initw, imgsz=32,
                          seed_filters=seed_filters, init_option=args.init_option)
elif args.model == 'nlr101':
    net = NLGroupResNet101(nlconv_outchans=args.nlcvoutch, groups=args.groups, shuffle=args.shuffle,
                           nlcv=args.nlcv, nlf=(args.nlf1, args.nlf2), init_weights=args.initw, imgsz=32,
                           seed_filters=seed_filters, init_option=args.init_option)
else:
    raise NotImplementedError(f'model {args.model} not implemented.')
# print(net)

if hasattr(net, "seed_filters"):
    # net.print_seed_filters()
    # net.print_nlconv_weights()
    # net.print_nlconv_size()
    pass

# model parameters evaluation
# from torchprofile import profile_macs
# dt = torch.rand(1, 3, 32, 32)
# flops = profile_macs(net, dt) / 1e6
# print("Mac is {}".format(flops))
# from torchinfo import summary
# summary(net, (1, 3, 32, 32), depth=10)

# ----------TRAINING----------
print('----------TRAINING----------')
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# train(net)
global best_acc
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# prepare datasets
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root=args.data, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root=args.data, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net = net.to(device)

# training optimizor, scheduler and loss function
criterion = nn.CrossEntropyLoss()


# filter out parameters with and without weight decay
def add_weight_decay(model, weight_decay=5e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


# this will remove BN parameters from weight decay
parameters = add_weight_decay(net, args.wd, skip_list=('linear',))
weight_decay = 0.  # override the weight decay value

optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=weight_decay)
if args.sche == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
else:
    raise NotImplementedError

# checkpoint saving
save_dir = args.output
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

w_dir = save_dir + '/weights'  # weights dir
if not os.path.exists(w_dir):  # make dir
    os.makedirs(w_dir)

if args.model == 'nlr18' or args.model == 'nlr50' or args.model == 'nlr101':
    last, best = w_dir + '/{}_{}_{}_{}_{}_g{}_last.pth'.format(args.model, args.nlcv, args.nlf1,
                                                               args.nlf2, args.nlcvoutch, args.groups),\
                 w_dir + '/{}_{}_{}_{}_{}_g{}_best.pth'.format(args.model, args.nlcv, args.nlf1,
                                                               args.nlf2, args.nlcvoutch, args.groups)
else:
    last, best = w_dir + '/{}_last.pth'.format(args.model), w_dir + '/{}_best.pth'.format(args.model)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.parameters()), max_norm=5)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print('Train Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if torch.isnan(torch.tensor(loss.item())):
            return True

    return False


def test(net, testloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    correct_sample_indices = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct_sample_indices.extend(predicted.eq(targets).cpu().detach().numpy().tolist())
            correct += predicted.eq(targets).sum().item()

    print('Test Loss: {:.3f} | Acc: {:.3f} ({:d}/{:d})'.format(
        test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100. * correct / total, np.where(correct_sample_indices)[0]


# training
for epoch in range(start_epoch, start_epoch+args.epochs):
    terminate = train(epoch)
    acc, sampler_indices = test(net, testloader)
    if best_acc < acc:
        best_acc = acc
        print('Save best model. acc: {:.3f} in epoch {}.'.format(best_acc, epoch))
        torch.save(net.state_dict(), best)
    print('Save last model in epoch {}.'.format(epoch))
    torch.save(net.state_dict(), last)

    scheduler.step()

    if terminate:
        break

print('Best acc: {:.3f}'.format(best_acc))

# ----------TRANSFORMATION----------
print('----------TRANSFORMATION----------')
# pytorch transform to onnx
net = net.to('cpu')
net.eval()

x = torch.randn([1, 3, 32, 32])

if args.model == 'nlr18' or args.model == 'nlr50' or args.model == 'nlr101':
    # intermediate model
    onnx_save_path = save_dir + '/{}_{}_{}_{}_{}_g{}_interm.onnx'.format(args.model, args.nlcv, args.nlf1,
                                                                         args.nlf2, args.nlcvoutch, args.groups)

    torch.onnx.export(net, x, onnx_save_path)
    # surgical model
    onnx_surgery_path = save_dir + '/{}_{}_{}_{}_{}_g{}_surgery.onnx'.format(args.model, args.nlcv, args.nlf1,
                                                                             args.nlf2, args.nlcvoutch, args.groups)

    # ----------SURGERY----------
    print('----------SURGERY----------')
    # onnx surgery
    onnxsurg = Surgery(onnx_save_path)

    # replace weight of NL conv in onnx graph with empty weight (0-Dimension tensor) as placeholder.
    if args.model == 'nlr18':
        # NL Resnet18 [2, 2, 2, 2]
        gen = [2, 2, 2, 2]
        repeat = 2
        # 1.0->1.1; 2.0->2.1; 3.0->3.1; 4.0->4.1
        node_idx2_rule = [0, 3] * 8  # 16/2=8
    elif args.model == 'nlr50':
        # NL Resnet50 [3, 4, 6, 3]
        gen = [3, 4, 6, 3]
        repeat = 3
        # 1.0->1.2; 2.0->2.3; 3.0->3.3; 4.0->4.2
        node_idx2_rule = [0, 3, 6] * 16  # 48/3=16
    else:  # nlr101
        # NL Resnet101 [3, 4, 23, 3]
        gen = [3, 4, 23, 3]
        repeat = 3
        # 1.0->1.2; 2.0->2.3; 3.0->3.22; 4.0->4.2
        node_idx2_rule = [0, 3, 6] * 33  # 99/3=33

    big_list = [[] for _ in range(len(gen))]
    for i in range(len(gen)):
        for g in range(gen[i]):
            for r in range(repeat):
                big_list[i].append(float('{}.{}'.format(i + 1, g)))
    node_idx1_rule = [i for item in big_list for i in item]

    for idx in range(nl_layers_num):
        onnxsurg.set_weight_by_name('layer{}.left.{}.nllayer.NLconv.weight'.format(f'{node_idx1_rule[idx]:.1f}',
                                                                                   node_idx2_rule[idx]))

    onnxsurg.export(onnx_surgery_path)

else:
    # resnet18 and resnet50
    onnx_save_path = save_dir + '/{}_transm.onnx'.format(args.model)

    torch.onnx.export(net, x, onnx_save_path)
