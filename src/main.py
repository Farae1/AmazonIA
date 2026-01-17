import os, glob
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from random import choice

# Importações dos seus outros arquivos
from dataset import GeoTiffDataset
from model import UNet
from train import train_epoch, validate_epoch
from metrics import iou_score, dice_score
from evaluate import predict_mask, evaluate_prediction


# (A função visualize_prediction foi movida para este arquivo)

# ========================
# Função de Visualização
# ========================
def visualize_prediction(model, dataset, device, idx=0):
    model.eval()
    img, mask = dataset[idx]  # img agora é (4, H, W)

    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).squeeze().cpu().numpy()

    img_np = np.transpose(img.numpy(), (1, 2, 0))  # Shape (H, W, 4)
    mask_np = mask.numpy()

    #    Assumindo que a ordem é (R, G, B, NIR)
    #    Se for BGR, mude para img_np[:, :, [2, 1, 0]]
    img_to_show = img_np[:, :, :3]

    # Garante que os valores estão no range [0, 1] para imshow
    img_to_show = np.clip(img_to_show, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_to_show);
    axs[0].set_title("Imagem (RGB)")
    axs[1].imshow(mask_np, cmap="gray");
    axs[1].set_title("Máscara real")
    axs[2].imshow(pred, cmap="gray");
    axs[2].set_title("Predição")
    for ax in axs: ax.axis("off")
    plt.show()


# ========================
# Configurações
# ========================
image_dir = "data/images/"
mask_dir = "data/masks/"
batch_size = 4
epochs = 100
lr = 1e-4

# ========================
# Dataset
# ========================
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

# Divide em treino/validação/teste (60/20/20)
train_imgs, test_imgs, train_masks, test_masks = train_test_split(image_paths, mask_paths, test_size=0.2,
                                                                  random_state=42)
train_imgs, val_imgs, train_masks, val_masks = train_test_split(train_imgs, train_masks, test_size=0.25,
                                                                random_state=42)

train_dataset = GeoTiffDataset(train_imgs, train_masks)
val_dataset = GeoTiffDataset(val_imgs, val_masks)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ========================
# Modelo
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=2).to(device)  # <- Isto agora usa a U-Net de 4 canais
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
metrics = {"iou": iou_score, "dice": dice_score}

# ========================
# Importe de pesos
# ========================
# Descomente se você já tem um modelo treinado E quer PULAR o treino
model.load_state_dict(torch.load("unet_deforestation_4band.pth", map_location=device))

# ========================
# Treinamento
# ========================
# Comente esta seção se você carregou pesos e quer apenas testar
#print("Iniciando treinamento...")
#for epoch in range(epochs):
#    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
#    val_loss, val_iou, val_dice = validate_epoch(model, val_loader, criterion, metrics, device)
#    print(
#        f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | IoU: {val_iou:.3f} | Dice: {val_dice:.3f}")

#torch.save(model.state_dict(), "unet_deforestation_4band.pth")
#print("Modelo salvo em unet_deforestation_4band.pth")

# ========================
# Teste em imagens novas
# ========================
print("\n=== Avaliação no conjunto de teste ===")
all_acc, all_iou, all_dice = [], [], []

for img_path, mask_path in zip(test_imgs, test_masks):
    # 🔹 MUDANÇA: Passa o 'device' para a função de predição
    pred_mask = predict_mask(model, img_path, device)
    with rasterio.open(mask_path) as src:
        true_mask = src.read(1).astype(np.uint8)
        true_mask = np.where(true_mask > 0, 1, 0)  # Garante que é binária

    m = evaluate_prediction(pred_mask, true_mask)
    all_acc.append(m["accuracy"])
    all_iou.append(m["iou"])
    all_dice.append(m["dice"])

print(f"Acurácia média: {np.mean(all_acc):.3f}")
print(f"IoU médio: {np.mean(all_iou):.3f}")
print(f"Dice médio: {np.mean(all_dice):.3f}")

# ========================
# Visualização de exemplo no teste
# ========================
print("\n=== Visualizando exemplo de Teste ===")
# MUDANÇA: Cria um Test Dataset para facilitar a visualização
test_dataset = GeoTiffDataset(test_imgs, test_masks)
idx = choice(range(len(test_dataset)))

# MUDANÇA: Chama a função de visualização com o dataset de TESTE
visualize_prediction(model, test_dataset, device, idx=idx)
