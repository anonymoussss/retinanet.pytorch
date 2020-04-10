import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import one_hot_embedding
## https://github.com/c0nn3r/RetinaNet/blob/master/focal_loss.py

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, balance_alpha = 0.25, num_classes = 20):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.balance_alpha = balance_alpha
        self.num_classes = num_classes
    def focal_loss(self, pred, target):
        '''Focal loss.
        Args:
          pred: (tensor) sized [N,C].C is number of classes
          target: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        not used this time
        '''
        log_pt = - F.cross_entropy(pred, target) # CE=-log(pt) based on the paper
        pt = torch.exp(log_pt)
        focal_loss = -((1 - pt) ** self.gamma) * log_pt
        balanced_focal_loss = self.balance_alpha * focal_loss
        return balanced_focal_loss
    def focal_loss_alt(self, x, y):
        # https://github.com/kuangliu/pytorch-retinanet/issues/52  @miramind
        """Focal loss
        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
        Returns:
            (tensor): focal loss
        """
        #print(y)
        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes) # [N,21]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        logit = F.softmax(x)
        logit = logit.clamp(1e-7, 1.-1e-7)
        conf_loss_tmp = -1 * t.float() * torch.log(logit)
        conf_loss_tmp = self.balance_alpha * conf_loss_tmp * (1-logit)**self.gamma
        conf_loss = conf_loss_tmp.sum()

        return conf_loss

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''

        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()
        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])
        ###
        num_pos = max(1.0, num_pos.item())
        ###
        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.item()/num_pos, cls_loss.item()/num_pos), end=' | ')
        loss = loc_loss / num_pos + cls_loss / num_pos
        return loss
def test_focal_loss():
    loss = FocalLoss()

    input = Variable(torch.randn(3, 5), requires_grad=True)
    target = Variable(torch.LongTensor(3).random_(5))

    print(input)
    print(target)

    output = loss(input, target)
    print(output)
    output.backward()