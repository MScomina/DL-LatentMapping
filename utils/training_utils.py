import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
import torchvision
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def get_dataset(dataset_name : str, train : bool = True, transform=None) -> Dataset:
    match dataset_name.lower():
        case "mnist":
            if train:
                dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            else:
                dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            expected_output_dim = (1, 28, 28)
        case "fashionmnist" | "fmnist":
            if train:
                dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            else:
                dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
            expected_output_dim = (1, 28, 28)
        case "cifar10":
            if train:
                dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            else:
                dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            expected_output_dim = (3, 32, 32)
        case "svhn":
            if train:
                dataset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform)
            else:
                dataset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform)
            expected_output_dim = (3, 32, 32)
        case "kmnist":
            if train:
                dataset = torchvision.datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
            else:
                dataset = torchvision.datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
            expected_output_dim = (1, 28, 28)
        case _:
            raise ValueError("Dataset not found")
    return dataset, expected_output_dim

def training_loop(model : nn.Module, dataloaders : tuple[DataLoader, DataLoader], criterion, optimizer, epochs : int, scheduler=None, save_name="model", device=device, should_save=False) -> nn.Module:
    train_loader, val_loader = dataloaders
    best_loss = float('inf')
    for epoch in range(epochs):  # loop over the dataset multiple times
        i = 0
        for data in train_loader:

            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            i += 1
            if i % 20 == 19:
                val_loss = test_loss(model, val_loader, criterion, device=device, should_print=False, use_subset=True)
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, val_loss))
        if scheduler is not None:
            scheduler.step()
        val_loss = test_loss(model, val_loader, criterion, device=device, should_print=False)
        if should_save and val_loss < best_loss:
            torch.save(model.state_dict(), "models/"+save_name+".pt")
            best_loss = val_loss
    
    return model

def test_loss(model : nn.Module, dataloader : DataLoader, criterion, device=device, should_print=True, use_subset=False) -> float:
    model.eval()
    total_loss = 0.0
    subset_size = 0.2  # Use 20% of the data if use_subset is True
    num_batches = int(len(dataloader) * subset_size) if use_subset else len(dataloader)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= num_batches:
                break
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item()
    avg_loss = total_loss / num_batches
    if should_print:
        print(f"Test loss: {avg_loss}")
    return avg_loss

def dataloader_generator(dataset, device, batch_size, shuffle, transform=None, num_workers=0) -> DataLoader:
    tensor_dataset = to_tensordataset(dataset, device, transform=transform)
    output = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return output

def to_tensordataset(dataset, device=device, transform=None) -> TensorDataset:
    data = []
    targets = []
    for image, label in dataset:
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        # Convert the image data to a PIL Image
        image = Image.fromarray(image)
        # Apply the transform to the image
        if transform is not None:
            image = transform(image)
        # Add the transformed image to the list
        data.append(image)
        # Add the label to the list
        targets.append(label)
    # Convert the list of tensors to a single tensor
    data = torch.stack(data).float().to(device)
    targets = torch.Tensor(targets).long().to(device)
    output = TensorDataset(data, targets)
    return output