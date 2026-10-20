import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """Expects preds to be probabilities (after sigmoid) and targets to be binary masks."""
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()

        dice = ((2. * intersection + self.smooth) /
               (preds.sum() + targets.sum() + self.smooth))

        return 1 - dice
    
class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        probs = torch.sigmoid(preds)
        dice_loss = self.dice(probs, targets)

        return {
            "bce_loss": bce_loss,
            "dice_loss": dice_loss,
            "total_loss": bce_loss + dice_loss
        }
    
def get_loss_function(config):
    if config["loss"]["name"] == "bce_dice":
        return BCEDiceLoss()
    elif config["loss"]["name"] == "dice":
        return DiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {config['loss']['name']}, Options are: ['bce_dice', 'dice']")
