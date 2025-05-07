import torch
import torch.nn.functional as F
import torch.nn as nn

# ce_loss_fn = nn.CrossEntropyLoss()

def dice_loss_per_slice(prob, label_1hot, smooth=1e-8):
    """
    Compute Dice loss slice-by-slice along the num_slice dimension.

    Args:
        prob (torch.Tensor): Predicted probabilities, shape [batch, Width, Height, num_slice, segt_class].
        label_1hot (torch.Tensor): One-hot encoded ground truth, shape [batch, Width, Height, num_slice, segt_class].
        smooth (float): Smoothing factor to prevent division by zero.

    Returns:
        losses_per_slice (torch.Tensor): Dice loss per slice, shape [num_slice].
        avg_loss (torch.Tensor): Average Dice loss across all slices.
    """
    # Get the num_slice dimension
    num_slice = prob.shape[3]

    # Container for per-slice Dice losses
    losses_per_slice = []

    # Loop through each slice
    for i in range(num_slice):
        # Extract slice i
        prob_slice = prob[:, :, :, i, :]
        label_slice = label_1hot[:, :, :, i, :]

        # Compute intersection and total for the slice
        intersection = torch.sum(prob_slice * label_slice, dim=[0, 1, 2])  # Sum over batch and spatial dimensions
        total = torch.sum(prob_slice + label_slice, dim=[0, 1, 2])

        # Compute Dice score for this slice
        dice_score = (2 * intersection + smooth) / (total + smooth)
        loss = 1 - dice_score.mean()  # Average Dice loss for this slice

        losses_per_slice.append(loss)

    # Convert to tensor
    losses_per_slice = torch.stack(losses_per_slice)

    # Average Dice loss across slices
    avg_loss = losses_per_slice.mean()

    return avg_loss

# Dice loss for 3D data in PyTorch
def dice_loss_3D(prob, label_1hot, smooth=1e-8):
    intersection = torch.sum(prob * label_1hot, dim=[0, 1, 2, 3])
    total = torch.sum(prob + label_1hot, dim=[0, 1, 2, 3])
    dice_score = (2 * intersection + smooth) / (total + smooth)
    loss = 1 - dice_score.mean()
    return loss

# Accuracy and Dice score calculation for 3D data
def torch_loss_accuracy_3D(logits_segt, segt_pl, segt_class):
    prob_segt = F.softmax(logits_segt, dim=-1)  # shape: [batch, Width, Height, n_slice, segt_class]
    pred_segt = torch.argmax(prob_segt, dim=-1)  # shape: [batch, Width, Height, n_slice]
    
    # label_1hot = F.one_hot(segt_pl.long(), num_classes=segt_class).permute(0, 1, 2, 3, 4).float()
    label_1hot = F.one_hot(segt_pl.long(), num_classes=segt_class).float()
    # pred_1hot = F.one_hot(pred_segt.long(), num_classes=segt_class).permute(0, 1, 2, 3, 4).float()
    pred_1hot = F.one_hot(pred_segt.long(), num_classes=segt_class).float()

    # ce_loss = ce_loss_fn(logits_segt.permute(0, 4, 1, 2, 3), segt_pl.long()) 
    # print(ce_loss.shape)
    # exit()

    loss_segt = dice_loss_per_slice(prob_segt, label_1hot)
    accuracy_segt = categorical_accuracy(pred_segt, segt_pl)
    dice_all = dice_3D(pred_1hot, label_1hot)
    
    dice_0 = dice_3D_single(pred_1hot[:, 0], label_1hot[:, 0])
    dice_1 = dice_3D_single(pred_1hot[:, 1], label_1hot[:, 1])
    dice_2 = dice_3D_single(pred_1hot[:, 2], label_1hot[:, 2])
    dice_3 = dice_3D_single(pred_1hot[:, 3], label_1hot[:, 3])

    return loss_segt, accuracy_segt, dice_0, dice_1, dice_2, dice_3, dice_all, pred_segt

# Dice loss for 2D data (channel-first format)
def dice_loss_2D_channel_first(prob, label_1hot, smooth=1e-8):
    intersection = torch.sum(prob * label_1hot, dim=[0, 2, 3])
    total = torch.sum(prob + label_1hot, dim=[0, 2, 3])
    dice_score = (2 * intersection + smooth) / (total + smooth)
    loss = 1 - dice_score.mean()
    return loss

# 2D Dice score calculation
def dice_2D_cf(pred_1hot, label_1hot, smooth=1e-8):
    intersection = torch.sum(pred_1hot * label_1hot, dim=[0, 2, 3])
    total = torch.sum(pred_1hot + label_1hot, dim=[0, 2, 3])
    dice_score = (2 * intersection + smooth) / (total + smooth)
    return dice_score.mean()

def dice_2D_cf_single(pred_1hot, label_1hot, smooth=1e-8):
    intersection = torch.sum(pred_1hot * label_1hot, dim=[0, 1, 2])
    total = torch.sum(pred_1hot + label_1hot, dim=[0, 1, 2])
    dice_score = (2 * intersection + smooth) / (total + smooth)
    return dice_score.mean()

# 3D Dice score calculation
def dice_3D(pred_1hot, label_1hot, smooth=1e-8):
    intersection = torch.sum(pred_1hot * label_1hot, dim=[0, 1, 2, 3])
    total = torch.sum(pred_1hot + label_1hot, dim=[0, 1, 2, 3])
    dice_score = (2 * intersection + smooth) / (total + smooth)
    return dice_score.mean()

def dice_3D_single(pred_1hot, label_1hot, smooth=1e-8):
    intersection = torch.sum(pred_1hot * label_1hot, dim=[0, 1, 2])
    total = torch.sum(pred_1hot + label_1hot, dim=[0, 1, 2])
    dice_score = (2 * intersection + smooth) / (total + smooth)
    return dice_score.mean()

# Categorical accuracy function
def categorical_accuracy(pred, truth):
    # correct = (pred == truth).float()
    # accuracy = correct.mean().item()
    # return accuracy
    return torch.mean((pred == truth).float())