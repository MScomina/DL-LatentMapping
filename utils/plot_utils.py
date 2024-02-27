import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader

# Test Examples
should_save = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_examples(model : nn.Module, dataloader : DataLoader, should_save=should_save, save_name="", device=device):
    model.eval()
    dataiter = iter(dataloader)
    data = next(dataiter)
    images, labels = data
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    images = images[:10]
    outputs = outputs[:10]
    images = images/2 + 0.5
    outputs = outputs/2 + 0.5
    fig, axs = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        if images[i].shape[0] == 1 or images[i].shape[-1] == 1:
            # Grayscale image, no need to transpose
            axs[0, i].imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
            axs[1, i].imshow(outputs[i].detach().cpu().numpy().squeeze(), cmap='gray')
        else:
            # RGB image, transpose from (num_channels, height, width) to (height, width, num_channels)
            axs[0, i].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
            axs[1, i].imshow(np.transpose(outputs[i].detach().cpu().numpy(), (1, 2, 0)))
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.tight_layout()
    if should_save:
        plt.savefig('plots/'+save_name+'.png')
    else:
        plt.show()
    plt.close(fig)