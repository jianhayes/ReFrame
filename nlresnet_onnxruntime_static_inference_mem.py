""" ReFrame: A Resource-Friendly Cloud-Assisted On-Device Deep Learning Framework for Vision Services
Author: Xie Jianhang
Github: https://github.com/jianhayes
Email: xiejianhang@bjtu.edu.cn; jianhang.xie@my.cityu.edu.hk
"""

import numpy as np
import onnxruntime as ort
import argparse
import os
import torch
import torchvision
import torchvision.transforms as transforms
import time
import psutil

parser = argparse.ArgumentParser("Non-Learnable ResNet inference in IoT device")
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

# checkpoint saving
save_dir = args.input
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if args.model == 'nlr18' or args.model == 'nlr50' or args.model == 'nlr101':
    onnx_deploy_path = save_dir + '/{}_{}_{}_{}_{}_g{}_deployment.onnx'.format(args.model, args.nlcv, args.nlf1,
                                                                               args.nlf2, args.nlcvoutch, args.groups)
else:  # resnet-18/-50/-101
    onnx_deploy_path = save_dir + '/{}_transm.onnx'.format(args.model)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

cpu_usage = process.cpu_percent()
mem = int(process.memory_info().rss / 1024 / 1024)
print("cpu count id {}.".format(cpu_count))
print("cpu usage percent is {}%".format(cpu_usage/cpu_count))
print("mem consumption is {} MB".format(mem))

# ----------INFERENCE----------
print('----------INFERENCE----------')
avg_cpu_usage = 0
avg_mem = 0
# sess = ort.InferenceSession(onnx_deploy_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
sess = ort.InferenceSession(onnx_deploy_path, providers=['CPUExecutionProvider'])

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

