import torch
import torch.nn as nn

from fpn import FPN50
from torch.autograd import Variable

class RetinaNet(nn.Module):
    num_anchors = 9 #A
    classes = 20 #K
    
    def __init__(self, num_classes=classes, fpn=FPN50()):
        super(RetinaNet,self).__init__()
        self.fpn = fpn
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)
        
    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    
    def forward(self,x):
        fms = self.fpn(x) # p3,p4,p5,p6,p7
        loc_preds = []
        cls_preds = []
        for fm in fms:
            
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
            #print(loc_pred.shape)
            #512测试中
            #torch.Size([2, 36864, 4])
            #torch.Size([2, 9216, 4])
            #torch.Size([2, 2304, 4])
            #torch.Size([2, 576, 4])
            #torch.Size([2, 144, 4])
        
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)
def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,512,512)))
    print(loc_preds.size()) #torch.Size([2, 49104, 4])
    print(cls_preds.size()) #torch.Size([2, 49104, 20])
    

#test()