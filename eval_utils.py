import torch

def get_mask(pred, threshold=0.5):
    """
    Convert raw model predictions (logits) into binary masks using a threshold.

    Args:
        pred (torch.Tensor): The raw prediction tensor with shape (B, 1, H, W), without sigmoid applied.
        threshold (float): Threshold to convert probabilities to binary values. Default is 0.5.

    Returns:
        torch.Tensor: Binary mask tensor with values 0 or 1, shape (B, 1, H, W).
    """
    prob = torch.sigmoid(pred)
    binary_mask = (prob >= threshold).float()

    return binary_mask

def confusion_matrix(pred, mask, threshold):
    """
    Compute TP, TN, FP, FN in batch. Return shape (B, 4).

    Args:
        pred_mask (torch.Tensor): Binary prediction mask, shape (B, 1, H, W)
        gt_mask (torch.Tensor): Binary ground truth mask, shape (B, 1, H, W)

    Returns:
        torch.Tensor: Tensor of shape (B, 4) where each row is [TP, TN, FP, FN]
    """
    
    pred_mask = get_mask(pred, threshold)
    assert pred_mask.shape == mask.shape, "Shape mismatch between prediction and ground truth."

    # Ensure masks are float tensors with binary values
    p = pred_mask.float().view(pred_mask.shape[0], -1)  # (B, H*W)
    m = mask.float().view(mask.shape[0], -1)        # (B, H*W)

    TP = ((p == 1) & (m == 1)).sum(dim=1)
    TN = ((p == 0) & (m == 0)).sum(dim=1)
    FP = ((p == 1) & (m == 0)).sum(dim=1)
    FN = ((p == 0) & (m == 1)).sum(dim=1)

    # Stack into shape (B, 4)
    return torch.stack([TP, TN, FP, FN], dim=1)

def compute_metrics(pred, mask, threshold=0.5, eps=1e-6):
    """
    Compute IoU (Jaccard), Dice Score, and Accuracy from confusion matrix.

    Args:
        confusion_matrix (torch.Tensor): Tensor of shape (B, 4), columns are [TP, TN, FP, FN]
        eps (float): Small constant to avoid division by zero

    Returns:
        torch.Tensor: Tensor of shape (B, 4), columns are [IoU, Dice, Accuracy, Sensitivity]
    """
    conf_mat = confusion_matrix(pred, mask, threshold=threshold)

    TP = conf_mat[:, 0]
    TN = conf_mat[:, 1]
    FP = conf_mat[:, 2]
    FN = conf_mat[:, 3]

    IoU = TP / (TP + FP + FN + eps)
    Dice = (2 * TP) / (2 * TP + FP + FN + eps)
    Accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    Sensitivity = TP / (TP + FN + eps)

    return torch.stack([IoU, Dice, Accuracy, Sensitivity], dim=1)

