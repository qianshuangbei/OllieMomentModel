import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class FocalLoss(nn.Module):
    def __init__(self, alpha=[0.8, 1.2, 0.8, 1.0, 0.8], gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
    
    def forward(self, pred, target):
        if pred.device != self.alpha.device:
            self.alpha = self.alpha.to(pred.device)
        
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred_softmax)
        target_one_hot.scatter_(1, target.view(-1, 1), 1)
        
        pt = (target_one_hot * pred_softmax).sum(1)
        batch_size = target.size(0)
        
        at = torch.zeros(batch_size, device=pred.device)
        for i in range(batch_size):
            at[i] = self.alpha[target[i]]
            
        focal_loss = -at * (1-pt).pow(self.gamma) * pt.log()
        return focal_loss.mean()

class MixedLoss(nn.Module):
    def __init__(self, config):
        super(MixedLoss, self).__init__()
        self.config = config
        
        if config['loss']['focal']['use']:
            self.focal = FocalLoss(
                alpha=config['loss']['focal']['alpha'],
                gamma=config['loss']['focal']['gamma']
            )
        
        if config['loss']['smoothing']['use']:
            self.smoothing = LabelSmoothingLoss(
                classes=config['training']['num_classes'],
                smoothing=config['loss']['smoothing']['smoothing']
            )
        
        self.focal_weight = config['loss']['loss_weights']['focal']
        self.smoothing_weight = config['loss']['loss_weights']['smoothing']
    
    def forward(self, pred, target):
        loss = 0
        if self.config['loss']['focal']['use']:
            loss += self.focal_weight * self.focal(pred, target)
        if self.config['loss']['smoothing']['use']:
            loss += self.smoothing_weight * self.smoothing(pred, target)
        return loss