import torch
import rasterio
import numpy as np
from sklearn.metrics import confusion_matrix


# --- Função para gerar máscara predita ---
def predict_mask(model, tif_path, device, threshold=0.5):
    model.eval()
    with rasterio.open(tif_path) as src:
        # 🔹 MUDANÇA: Lê TODAS as bandas (C, H, W)
        img = src.read().astype(np.float32)

    # 🔹 MUDANÇA: Normalização Min-Max por canal (IDÊNTICA ao Dataset)
    if img.max() > 1.0:
        for c in range(img.shape[0]):
            min_val = img[c].min()
            max_val = img[c].max()
            img[c] = (img[c] - min_val) / (max_val - min_val + 1e-8)

    # Adiciona dimensão de batch (1, C, H, W)
    img_tensor = torch.tensor(img).unsqueeze(0)

    with torch.no_grad():
        # 🔹 MUDANÇA: Mova o tensor para o 'device' (ex: GPU)
        pred = model(img_tensor.to(device))

    # Caso binário (1 canal) - (Mantido para flexibilidade, embora a U-Net tenha n_classes=2)
    if pred.shape[1] == 1:
        pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        binary_mask = (pred_mask > threshold).astype(np.uint8)

    # Caso multiclasses (n_classes=2)
    else:
        pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        binary_mask = pred_mask

    return binary_mask


# --- Função para comparar máscara predita x real ---
def evaluate_prediction(pred_mask, true_mask):
    pred = pred_mask.flatten()
    true = true_mask.flatten()

    # Garante que as labels [0, 1] existem para o ravel()
    try:
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[0, 1]).ravel()
    except ValueError:
        # Caso muito raro onde só existe uma classe na imagem
        if np.all(true == 0):
            tn, fp, fn, tp = len(true), 0, 0, 0
        elif np.all(true == 1):
            tn, fp, fn, tp = 0, 0, 0, len(true)
        else:
            return {"accuracy": 0, "iou": 0, "dice": 0}  # Caso inesperado

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

    return {"accuracy": accuracy, "iou": iou, "dice": dice}