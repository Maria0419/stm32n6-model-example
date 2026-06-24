import torch
from torch import nn


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=1.0, pos_weight=1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))

    def forward(self, logits, masks):
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * masks).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, masks, pos_weight=self.pos_weight
        )
        return dice_loss + self.bce_weight * bce_loss


class SquareSegModel(nn.Module):
    def __init__(self, channels=16, layers=7, kernel_size=3):
        super().__init__()

        blocks = []
        in_channels = 1
        padding = kernel_size // 2
        for _ in range(layers - 1):
            blocks.append(
                nn.Conv2d(in_channels, channels, kernel_size, padding=padding, bias=True)
            )
            blocks.append(nn.ReLU(inplace=False))
            in_channels = channels
        blocks.append(nn.Conv2d(channels, 1, kernel_size=1, bias=True))
        self.net = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.net(inputs)


def build_model(config):
    params = config.get("params", {})

    return SquareSegModel(**params)

