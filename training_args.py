import torch    
import torch.nn.functional as F

def Make_Optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def Make_LR_Scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10, eta_min = 1e-6)

def Make_Loss_Function(number_of_classes):
    class DiceCELoss:
        def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
            self.weight = weight
            self.epsilon = epsilon
            self.mode = mode
        
        def __call__(self, pred, target):
            if self.mode == 'binary':
                pred = pred.squeeze(1)  # shape: (batchsize, H, W)
                target = target.squeeze(1).float()
                intersection = torch.sum(pred * target, dim=(1, 2))
                union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()
                
                ce_loss = F.binary_cross_entropy(pred, target)
            
            elif self.mode == 'multiclass':
                device = "cuda" if torch.cuda.is_available() else "cpu"
                weight = torch.tensor([0.0083, 1.3227, 3.1356, 1.1233, 1.7000, 1.4327, 0.5643, 0.6851, 0.3903,
                                       0.9699, 0.9778, 0.7841, 0.5153, 0.9659, 0.8963, 0.1931, 1.6641, 1.1725,
                                       0.6847, 0.6206, 1.1934], dtype=torch.float32, device=device)
                batchsize, num_classes, H, W = pred.shape
                pred_soft = torch.softmax(pred, dim=1)
                target = target.squeeze(1)
                target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
                intersection = torch.sum(pred_soft * target_one_hot, dim=(2, 3))
                union = torch.sum(pred_soft, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
                dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
                dice_loss = 1 - dice.mean()
                
                ce_loss = F.cross_entropy(pred, target, weight=weight)
            else:
                raise ValueError("mode should be 'binary' or 'multiclass'")
            
            combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
            
            return combined_loss
    
    BINARY_SEG = True if number_of_classes==2 else False
    return DiceCELoss(mode='binary') if BINARY_SEG else DiceCELoss(mode='multiclass')
