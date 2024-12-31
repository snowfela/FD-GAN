import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    """
    Computes the adversarial loss for GAN training.
    """
    def __init__(self, mode='vanilla'):
        super(AdversarialLoss, self).__init__()
        self.mode = mode
        if mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()  # Combines sigmoid activation with BCE
        elif mode == 'hinge':
            self.loss = None  # Hinge loss doesn't use BCE
        else:
            raise ValueError(f"Unsupported loss mode: {mode}")

    def forward(self, logits, is_real):
        if self.mode == 'vanilla':
            labels = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
            return self.loss(logits, labels)
        elif self.mode == 'hinge':
            if is_real:
                return torch.mean(torch.relu(1.0 - logits))
            else:
                return torch.mean(torch.relu(1.0 + logits))


def classification_loss(predictions, labels):
    """
    Compute the binary classification loss using BCELoss (Binary Cross Entropy Loss).
    """
    criterion = torch.nn.BCELoss()
    return criterion(predictions, labels)

def forgery_similarity_loss(predicted_masks, ground_truth_masks):
    """
    Compute the forgery similarity loss, comparing predicted and ground truth masks.
    """
    return torch.mean(torch.abs(predicted_masks - ground_truth_masks))

def diversity_loss(mask_filters):
    """
    Compute the diversity loss between mask filters.
    """
    loss = 0
    for i in range(len(mask_filters)):
        for j in range(i+1, len(mask_filters)):
            loss += torch.cosine_similarity(mask_filters[i].weight.flatten(), mask_filters[j].weight.flatten(), dim=0)
    return loss
