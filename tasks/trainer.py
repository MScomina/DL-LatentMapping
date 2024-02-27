import torch
import torchvision.transforms as transforms
from torch import nn
import os

from models.visual_transformer import VisualTransformer
from models.autoencoder import AutoEncoder

from utils.plot_utils import print_examples
from utils.training_utils import test_loss, training_loop, dataloader_generator, get_dataset

# Default hyperparameters

# Dataset specific parameters
dataset_path = "cifar10"
expected_output_dim = (3, 32, 32)
rotations = 0
patch_size = 4
train_split = 0.85

# Training parameters
epochs = 100
batch_size = 128
lr = 1e-4
dropout = 0.0
weight_decay = 0
gamma_decay = 0.9
epochs_decay = 10

# Model parameters
load_model = True

# Autoencoder parameters
linear_layers = 3
autoencoder_save_path = "autoencoder"

# Transformer models parameters
nhead = 4
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 1024
transformer_save_path = "visual_transformer"

# Generative models parameters
mask_prob = 0.1
blank_probability = 0.05
label_dim = 10
noise_dim = 128

def autoencoder_model(latent_space_size = -1, dataset_name=dataset_path, batch_size = batch_size, epochs = epochs, lr = lr,
                      weight_decay = weight_decay, dropout = dropout, load_model = load_model, rotations = rotations, patch_size = patch_size, 
                      gamma_decay = gamma_decay, epochs_decay = epochs_decay, linear_layers = linear_layers, save_name = autoencoder_save_path):
    
    train_dataset, expected_output_dim = get_dataset(dataset_name, train=True)
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset, _ = get_dataset(dataset_name, train=False)
    
    if latent_space_size == -1:
        latent_space_size = expected_output_dim[0] * expected_output_dim[1] * expected_output_dim[2] // patch_size**2
    
    model = AutoEncoder(
        latent_space_size=latent_space_size, 
        expected_output_dim=expected_output_dim,
        dropout=dropout,
        linear_layers=linear_layers
    )

    if load_model and os.path.isfile("models/"+dataset_name+"/"+save_name+".pt"):
        model.load_state_dict(torch.load("models/"+dataset_name+"/"+save_name+".pt"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if expected_output_dim[0] == 1:
        transform_list = [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, ), (0.5, ))
            ]
    else:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    if rotations > 0:
        transform_list.insert(0, transforms.RandomRotation(rotations))
    transform = transforms.Compose(transform_list)

    trainloader = dataloader_generator(
        dataset=train_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        transform=transform
    )

    valloader = dataloader_generator(
        dataset=val_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        transform=transform
    )

    testloader = dataloader_generator(
        dataset=test_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        transform=transform
    )

    del train_dataset, val_dataset, test_dataset
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_decay, gamma=gamma_decay)

    model = training_loop(
        model=model, 
        dataloaders=(trainloader, valloader), 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        epochs=epochs, 
        save_name=dataset_name+"/"+save_name,
        device=device,
        should_save=True
    )

    test_loss(
        model=model, 
        dataloader=testloader, 
        criterion=criterion, 
        device=device
    )

    print_examples(
        model, 
        testloader, 
        should_save=True, 
        save_name=dataset_name+"/"+save_name,
        device=device
    )

    #torch.save(model.state_dict(), "models/"+dataset_name+"/"+save_name+".pt")

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


def visual_transformer_model(nhead = nhead, num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers, dim_feedforward = dim_feedforward,
                            dropout = dropout, batch_size = batch_size, epochs = epochs, lr = lr, dataset_name = dataset_path,
                            weight_decay = weight_decay, load_model = load_model, rotations = rotations, patch_size = patch_size, 
                            gamma_decay = gamma_decay, epochs_decay = epochs_decay, save_name = transformer_save_path):
    
    train_dataset, expected_output_dim = get_dataset(dataset_name, train=True)
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset, _ = get_dataset(dataset_name, train=False)

    model = VisualTransformer(
        nhead=nhead, 
        num_encoder_layers=num_encoder_layers, 
        num_decoder_layers=num_decoder_layers, 
        dim_feedforward=dim_feedforward, 
        dropout=dropout,
        patch_size=patch_size,
        expected_dimension=expected_output_dim
    )

    if load_model and os.path.isfile("models/"+dataset_path+"/"+save_name+".pt"):
        model.load_state_dict(torch.load("models/"+dataset_path+"/"+save_name+".pt"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if expected_output_dim[0] == 1:
        transform_list = [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, ), (0.5, ))
            ]
    else:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    
    if rotations > 0:
        transform_list.insert(0, transforms.RandomRotation(rotations))
    transform = transforms.Compose(transform_list)

    trainloader = dataloader_generator(
        dataset=train_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        transform=transform
    )

    valloader = dataloader_generator(
        dataset=val_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        transform=transform
    )

    testloader = dataloader_generator(
        dataset=test_dataset,
        device=device,
        batch_size=batch_size,
        shuffle=False,
        transform=transform
    )

    del train_dataset, val_dataset, test_dataset

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_decay, gamma=gamma_decay)

    model = training_loop(
        model=model, 
        dataloaders=(trainloader, valloader), 
        criterion=criterion, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        epochs=epochs, 
        save_name=dataset_name+"/"+save_name,
        device=device,
        should_save=True
    )

    test_loss(
        model=model, 
        dataloader=testloader, 
        criterion=criterion, 
        device=device
    )

    print_examples(
        model, 
        testloader, 
        should_save=True, 
        save_name=dataset_name+"/"+save_name, 
        device=device
    )
        
    #torch.save(model.state_dict(), "models/"+dataset_path+"/"+save_name+".pt")

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))