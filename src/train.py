import torch

# -------------------------
# Função de treino
# -------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)

        # 🔹 Garante que a saída tem o mesmo tamanho da máscara
        outputs = torch.nn.functional.interpolate(
            outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
        )

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


# -------------------------
# Função de validação
# -------------------------
def validate_epoch(model, dataloader, criterion, metrics, device):
    model.eval()
    total_loss, total_iou, total_dice = 0, 0, 0

    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            outputs = torch.nn.functional.interpolate(
                outputs, size=masks.shape[1:], mode='bilinear', align_corners=False
            )

            loss = criterion(outputs, masks)
            total_loss += loss.item()
            total_iou += metrics["iou"](outputs, masks)
            total_dice += metrics["dice"](outputs, masks)

    n = len(dataloader)
    return total_loss / n, total_iou / n, total_dice / n
