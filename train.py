from __future__ import print_function

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import fpn
from loss import FocalLoss
from retinanet import RetinaNet
from datagen_xml import ListDataset
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--pretrained',required=True,
                        metavar="/path/to/weights.pth",
                        help='choose pretrained model as backbone')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(root="E:/数据集/pascal/VOC2012/JPEGImages",
    trainval_txt_path="E:/数据集/pascal/VOC2012/ImageSets/Main/train.txt",
    xml_path='E:/数据集/pascal/VOC2012/Annotations', train=True, transform=transform, input_size=512)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

testset = ListDataset(root="E:/数据集/pascal/VOC2012/JPEGImages",
    trainval_txt_path="E:/数据集/pascal/VOC2012/ImageSets/Main/val.txt",
    xml_path='E:/数据集/pascal/VOC2012/Annotations', train=False, transform=transform, input_size=512)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)

# FPN backbone
# Select pretrained pth to load
fpn_backbone = fpn.FPN50()
if args.pretrained == "res50":
    print("Using the resnet50 as backbone.")
    fpn_backbone = fpn.FPN50()
    net_dict = fpn_backbone.state_dict()
    pretrained_dict = torch.load('model/resnet50-19c8e357.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict} 
    #取出预训练模型中与新模型的dict中重合的部分
    net_dict.update(pretrained_dict)#用预训练模型参数更新new_model中的部分参数
    fpn_backbone.load_state_dict(net_dict) #将更新后的model_dict加载进new model中
elif args.pretrained == "res101":
    print("Using the resnet101 as backbone.")
    fpn_backbone = fpn.FPN101()
    net_dict = fpn_backbone.state_dict()
    pretrained_dict = torch.load('model/resnet101-5d3b4d8f.pth')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict} 
    net_dict.update(pretrained_dict)
    fpn_backbone.load_state_dict(net_dict) 
else:
    #使用resnet50从头训练
    print("Training from scratch")


# Model
net = RetinaNet(fpn=fpn_backbone)
args.resume=False
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets, fname) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        print("epoch: %d" %(epoch), end=' | ')
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.item(), train_loss/(batch_idx+1)))
   

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets,_) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        print("epoch: %d" % (epoch), end=' | ')
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.item()
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.item(), test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        print("best loss now : %.3f"%(test_loss))
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+50):
    train(epoch)
    test(epoch)