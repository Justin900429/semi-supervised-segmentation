from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class mIoULoss(torch.nn.Module):
    def __init__(self, num_classes: int):
        super(mIoULoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N = inputs.shape[0]
        inputs = inputs.log_softmax(dim=1).exp()

        inter = inputs * target
        inter = inter.view(N, self.num_classes, -1).sum(2)

        union = inputs + target - (inputs * target)
        union = union.view(N, self.num_classes, -1).sum(2)

        loss = 1 - inter / (union + 1e-8)

        return torch.mean(loss)


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 0.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        N = inputs.shape[0]
        inputs = inputs.log_softmax(dim=1).exp()

        inter = inputs * target
        inter = inter.view(N, self.num_classes, -1).sum(2)

        union = inputs + target
        union = union.view(N, self.num_classes, -1).sum(2)

        loss = 1 - (2 * inter + self.smooth) / (union + self.smooth).clamp_min(1e-8)

        return torch.mean(loss)


class SegmentLoss(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        threshold: float = 0.95,
        unlabel_weight: float = 1.0,
        temperature: float = 1.0,
        use_ssl: bool = True,
    ):
        super(SegmentLoss, self).__init__()
        self.catego_loss = partial(sigmoid_focal_loss, reduction="mean")
        self.miou_loss = mIoULoss(num_classes=num_classes)
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.threshold = threshold
        self.unlabel_weight = unlabel_weight
        self.temperature = temperature
        self.use_ssl = use_ssl

    def forward(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        strong_pred: torch.Tensor = None,
        weak_pred: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        # Labeled loss
        catego_loss = self.catego_loss(inputs, target.float())
        dice_loss = self.dice_loss(inputs, target.float())
        miou_loss = self.miou_loss(inputs, target.float())
        # Unlabeled loss
        distillation_loss = 0
        if weak_pred is not None:
            psuedo_prob = F.softmax(weak_pred / self.temperature, dim=1)
            max_prob_each_channel = psuedo_prob.amax(dim=1)
            mask = max_prob_each_channel > self.threshold
            if mask.sum() > 0:
                distillation_loss = (
                    F.kl_div(
                        F.log_softmax(strong_pred / self.temperature, dim=1),
                        F.log_softmax(weak_pred.detach() / self.temperature, dim=1),
                        reduction="none",
                        log_target=True,
                    )
                    * mask.unsqueeze(1)
                    * (self.temperature * self.temperature)
                ).sum() / mask.sum()
            else:
                distillation_loss = (strong_pred * 0).sum()

        total_loss = (
            catego_loss
            + dice_loss
            + miou_loss
            + self.unlabel_weight * distillation_loss
        )
        return {
            "catego_loss": catego_loss,
            "miou_loss": miou_loss,
            "dice_loss": dice_loss,
            "dis_loss": distillation_loss,
            "total_loss": total_loss,
        }
