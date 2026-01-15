import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_prediction(model, dataset, device, idx=0):
    model.eval()
    img, mask = dataset[idx]
    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    img_np = np.transpose(img.numpy(), (1,2,0))
    mask_np = mask.numpy()

    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(img_np); axs[0].set_title("Imagem")
    axs[1].imshow(mask_np, cmap="gray"); axs[1].set_title("Máscara real")
    axs[2].imshow(pred, cmap="gray"); axs[2].set_title("Predição")
    for ax in axs: ax.axis("off")
    plt.show()
