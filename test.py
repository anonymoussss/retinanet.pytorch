import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image,ImageDraw
import os
the_classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('checkpoint/ckpt.pth')['net'])
net.eval()
#net.cuda() #是否用GPU测试
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
#the_classes=[c.strip() for c in open('data/voc.names').readlines()]
encoder = DataEncoder()
for i in os.listdir('image'):
    print('Loading image %s...'%i)
    img = Image.open(os.path.join('image',i))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    w = h = 512
    img = img.resize((w,h))
    # img.save(os.path.join('output', i))
    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    x = Variable(x, volatile=True)
    #x =x.cuda() #是否用GPU测试
    loc_preds, cls_preds = net(x)
    loc_preds = loc_preds.squeeze()
    cls_preds = cls_preds.squeeze()
    print('Decoding..')
    
    boxes, labels,scores = encoder.decode(loc_preds.data, cls_preds.data, (w,h))
    #boxes=[],labels=[],scores=[]
    draw = ImageDraw.Draw(img)
    for index,box in enumerate(boxes):
        draw.rectangle(list(box), outline='red')
        #draw.text(list(box)[:2],text=the_classes[int(labels[index])])
        #print(int(labels[index]))
    img.save(os.path.join('output',i))