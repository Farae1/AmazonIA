import torch

def iou_score(outputs, masks, eps=1e-6):
    preds = outputs.argmax(dim=1).float()
    masks = masks.float()
    inter = (preds * masks).sum(dim=(1,2))
    union = preds.sum(dim=(1,2)) + masks.sum(dim=(1,2)) - inter
    return ((inter+eps) / (union+eps)).mean().item()

def dice_score(outputs, masks, eps=1e-6):
    preds = outputs.argmax(dim=1).float()
    masks = masks.float()
    inter = (preds * masks).sum(dim=(1,2))
    return ((2*inter+eps) / (preds.sum(dim=(1,2)) + masks.sum(dim=(1,2)) + eps)).mean().item()
