""" ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: xiejianhang@bjtu.edu.cn; jianhang.xie@my.cityu.edu.hk
"""

import numpy as np
import onnxruntime as ort
import argparse
import math
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import random
import onnx
from utils.onnx_surgery import Surgery
import psutil


def seed_xavier_normal_(tensor, gain=1., generator=None):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    with torch.no_grad():
        return tensor.normal_(0., std, generator=generator)


def seed_kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu', generator=None):
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)


def seed_std_normal_(tensor, mean=0., std=1., generator=None):
    return tensor.normal_(mean, std, generator=generator)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser("Non-Learnable ResNet dynamic inference in IoT device")
parser.add_argument('--seed', default=0, type=int, help='global random seed (training random seed)')
parser.add_argument('--nl-seed', default=1, type=int, help='non-learnable base random seed (init NL conv random seed)')
parser.add_argument('--mono-map', default='eqdiff', type=str, help='base seed monotonic mapping')
parser.add_argument('--init-option', default='kaiming_normal', type=str, help='NL parameters initialization function')
parser.add_argument('--model', default='nlr18', type=str, help='which model')
parser.add_argument('--nlcvoutch', default=-1, type=int, help='non-learnable conv output channels')
parser.add_argument('--input', default='./onnx_trasm',
                    type=str, help='name of the exchange dir to save onnx model')
parser.add_argument('--groups', default=4, type=int, help='pointwise convolution groups')
parser.add_argument('--nlcv', default='group', type=str, help='NL Conv')
parser.add_argument('--nlf1', default='max', type=str, help='NL filter1')
parser.add_argument('--nlf2', default='mix_avg', type=str, help='NL filter2')

parser.add_argument('--data', default='../../Datasets/CIFAR10', type=str, help='datasets path')
args = parser.parse_args()

# init psutil
process = psutil.Process(os.getpid())
cpu_count = psutil.cpu_count()
cpu_usage = process.cpu_percent()

# ----------GENERATION SETTINGS----------
print('----------GENERATION SETTINGS----------')
if args.seed == args.nl_seed:
    raise NotImplementedError(f'base seed {args.seed} and nl {args.nl_seed} seed should not be equal.')

# base seed
init_seed(args.seed)
print("base seed is {}".format(torch.initial_seed()))

nl_layers_num = 0
if args.model == 'nlr18':
    nl_layers_num = 16
elif args.model == 'nlr50':
    nl_layers_num = 48
elif args.model == 'nlr101':
    nl_layers_num = 99

# seed filters init
seed_filters = [0 for _ in range(nl_layers_num)]
if args.mono_map == 'const':
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
    seed_filters = None
else:
    raise NotImplementedError(f'seed generated function {args.seed_filter} not implemented.')

if args.seed in seed_filters:
    raise NotImplementedError(f'base seed {args.seed} should not in seed filters.')

print(seed_filters)

# checkpoint saving
save_dir = args.input
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cpu_usage = process.cpu_percent()
mem = int(process.memory_info().rss / 1024 / 1024)
print("cpu count id {}.".format(cpu_count))
print("before generation cpu usage percent is {}%".format(cpu_usage/cpu_count))
print("before generation mem consumption is {} MB".format(mem))

# ----------NL PARAMETERS DYNAMIC GENERATION IN MEMORY----------
print('----------NL PARAMETERS DYNAMIC GENERATION IN MEMORY----------')
# if nlcvoutch is auto and groups, shape is [inch, 1, 3, 3]
# elif nlcvoutch is auto and std, shape is [inch, inch, 3, 3]
# elif nlcvoutch is > 0 and groups, shape is [nlcvoutch, 1, 3, 3]
# elif nlcvoutch is > 0 and std, shape is [nlcvoutch, inch // gcd(inchans, nlcvoutch), 3, 3]

if args.model == 'nlr18' or args.model == 'nlr50' or args.model == 'nlr101':

    if args.model == 'nlr18':
        # ResNet18 32*32 [2, 2, 2, 2]
        gen = [2, 2, 2, 2]
        repeat = 2
        node_idx2_rule = [0, 3] * 8
        expansion = 1
    elif args.model == 'nlr50':
        # ResNet50 32*32 [3, 4, 6, 3]
        gen = [3, 4, 6, 3]
        repeat = 3
        node_idx2_rule = [0, 3, 6] * 16
        expansion = 4
    else:  # nlr101
        # ResNet101 32*32 [3, 4, 23, 3]
        gen = [3, 4, 23, 3]
        repeat = 3
        node_idx2_rule = [0, 3, 6] * 33
        expansion = 4

    in_planes = 64
    planes = [64, 128, 256, 512]

    big_inchans_list = [[] for _ in range(len(gen))]
    for i in range(len(gen)):
        print(i)
        for g in range(gen[i]):  # stage of gen
            big_inchans_list[i].append(int('{}'.format(in_planes)))
            for r in range(repeat - 1):  # stage of basicblock/bottlenet
                big_inchans_list[i].append(int('{}'.format(planes[i])))
            in_planes = planes[i] * expansion
    inchans = [i for item in big_inchans_list for i in item]

    if args.nlcvoutch == -1:
        outchans = inchans
    else:
        outchans = [args.nlcvoutch for _ in range(nl_layers_num)]

    big_list = [[] for _ in range(len(gen))]
    for i in range(len(gen)):
        for g in range(gen[i]):
            for r in range(repeat):
                big_list[i].append(float('{}.{}'.format(i + 1, g)))
    node_idx1_rule = [i for item in big_list for i in item]

    onnx_surgery_path = save_dir + '/{}_{}_{}_{}_{}_g{}_surgery.onnx'.format(args.model, args.nlcv, args.nlf1,
                                                                             args.nlf2, args.nlcvoutch, args.groups)
    onnxsurg = Surgery(onnx_surgery_path)

    avg_gen_cpu_usage = 0
    avg_gen_mem = 0

    for idx in range(nl_layers_num):
        g = torch.Generator()
        g.manual_seed(seed_filters[idx])
        if args.nlcv == 'std':
            shape = [outchans[idx], inchans[idx], 3, 3]
        elif args.nlcv == 'group':
            gcd = math.gcd(outchans[idx], inchans[idx])
            shape = [outchans[idx], inchans[idx] // gcd, 3, 3]
        else:
            raise NotImplementedError(f'{args.nlcv} not deploy.')
        W = torch.zeros(shape)
        if args.init_option == 'kaiming_normal':
            seed_kaiming_normal_(W, mode='fan_out', nonlinearity='relu', generator=g)
        elif args.init_option == 'xavier_normal':
            seed_xavier_normal_(W, generator=g)
        elif args.init_option == 'std_normal':
            seed_std_normal_(W, 0., 1., generator=g)
        else:
            raise NotImplementedError("only kaiming_normal/xavier_normal/std_normal init is supported")
        # print(W)

        onnxsurg.set_weight_by_name('layer{}.left.{}.nllayer.NLconv.weight'.format(f'{node_idx1_rule[idx]:.1f}',
                                                                                   node_idx2_rule[idx]),
                                    W.numpy().astype(np.float32))

        cpu_usage = process.cpu_percent()
        mem = int(process.memory_info().rss / 1024 / 1024)

        avg_gen_cpu_usage = avg_gen_cpu_usage + cpu_usage
        avg_gen_mem = avg_gen_mem + mem

    print("Average generation cpu usage percent is {}%".format(avg_gen_cpu_usage / (cpu_count * nl_layers_num)))
    print("Average generation mem consumption is {} MB".format(avg_gen_mem / nl_layers_num))

else:
    # others
    raise NotImplementedError(f'{args.model} not implemented or do not need generation.')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

# ----------DYNAMIC INFERENCE----------
print('----------DYNAMIC INFERENCE----------')
avg_cpu_usage = 0
avg_mem = 0
sess = ort.InferenceSession(onnx._serialize(onnxsurg.model), providers=['CPUExecutionProvider'])

# onnxruntime inference
start = time.time()

print("Start Testing!")
input_name = sess.get_inputs()[0].name
correct = 0
total = 0
for data in testloader:
    images, labels = data
    images, labels = images.numpy(), labels.numpy()
    # print(images.shape)
    outputs = sess.run(None, {input_name: images})

    cpu_usage = process.cpu_percent()
    mem = int(process.memory_info().rss / 1024 / 1024)

    predicted = np.argmax(outputs[0], axis=1)
    total += labels.shape[0]
    correct += (predicted == labels).sum()

    avg_cpu_usage = avg_cpu_usage + cpu_usage
    avg_mem = avg_mem + mem

acc = correct / total
print('ONNXRuntime Test Acc is: %.3f%%' % (100 * acc))

end = time.time()
print(f"Average response time cost: {1000 * (end - start) / len(testloader.dataset)} ms")

print("Average cpu usage percent is {}%".format(avg_cpu_usage/(cpu_count * 10000)))
print("Average mem consumption is {} MB".format(avg_mem / 10000))
