import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch import nn
import numpy as np


from models.visual_transformer import VisualTransformer
from models.autoencoder import AutoEncoder

from models.generative_transformer import GenerativeImageTransformer
from models.conditional_autoencoder import ConditionalAutoEncoder

# Default hyperparameters

# Dataset specific parameters
expected_output_dim = (3, 32, 32)
rotations = 0
patch_size = 4

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

# Transformer models parameters
nhead = 4
num_encoder_layers = 4
num_decoder_layers = 4
dim_feedforward = 1024

# Generative models parameters
mask_prob = 0.1
blank_probability = 0.05
label_dim = 10
noise_dim = 128


def autoencoder_model(latent_space_size = -1, expected_output_dim = expected_output_dim, batch_size = batch_size, epochs = epochs, lr = lr,
                      weight_decay = weight_decay, dropout = dropout, load_model = load_model, rotations = rotations, patch_size = patch_size, 
                      gamma_decay = gamma_decay, epochs_decay = epochs_decay, linear_layers = linear_layers):
    
    if latent_space_size == -1:
        latent_space_size = expected_output_dim[0] * expected_output_dim[1] * expected_output_dim[2] // patch_size**2
    
    model = AutoEncoder(
        latent_space_size=latent_space_size, 
        expected_output_dim=expected_output_dim,
        dropout=dropout,
        linear_layers=linear_layers
    )

    if load_model:
        model.load_state_dict(torch.load("models/autoencoder.pt"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform_list = [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    if rotations > 0:
        transform_list.insert(0, transforms.RandomRotation(rotations))
    transform = transforms.Compose(transform_list)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_decay, gamma=gamma_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        scheduler.step()

    k = False
    with torch.no_grad():

        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            print(f"Test loss: {loss.item()}")

            if not k:
                images = (images + 1) / 2
                outputs = (outputs + 1) / 2
                k = True
                fig, axs = plt.subplots(2, 10, figsize=(15, 3))
                for i in range(10):
                    axs[0, i].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
                    axs[0, i].axis('off')
                    axs[1, i].imshow(np.transpose(outputs[i].detach().cpu().numpy(), (1, 2, 0)))
                    axs[1, i].axis('off')
                plt.tight_layout()
                plt.savefig('plot_autoencoder.png')

    torch.save(model.state_dict(), "models/autoencoder.pt")


def visual_transformer_model(nhead = nhead, num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers, dim_feedforward = dim_feedforward,
                            dropout = dropout, expected_output_dim = expected_output_dim, batch_size = batch_size, epochs = epochs, lr = lr,
                            weight_decay = weight_decay, load_model = load_model, rotations = rotations, patch_size = patch_size, 
                            gamma_decay = gamma_decay, epochs_decay = epochs_decay):
    
    model = VisualTransformer(
        nhead, 
        num_encoder_layers, 
        num_decoder_layers, 
        dim_feedforward, 
        dropout,
        patch_size=patch_size,
        expected_dimension=expected_output_dim
    )

    if load_model:
        model.load_state_dict(torch.load("models/visual_transformer.pt"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform_list = [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    if rotations > 0:
        transform_list.insert(0, transforms.RandomRotation(rotations))
    transform = transforms.Compose(transform_list)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
                                                
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs_decay, gamma=gamma_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        scheduler.step()
        
    # Test the model and compare the input and output images
    k = False
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            print(f"Test loss: {loss.item()}")

            if not k:
                images = (images + 1) / 2
                outputs = (outputs + 1) / 2
                k = True
                fig, axs = plt.subplots(2, 10, figsize=(15, 3))
                for i in range(10):
                    axs[0, i].imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
                    axs[0, i].axis('off')
                    axs[1, i].imshow(np.transpose(outputs[i].detach().cpu().numpy(), (1, 2, 0)))
                    axs[1, i].axis('off')
                plt.tight_layout()
                plt.savefig('plot_visual_transformer.png')

    
        
    torch.save(model.state_dict(), "models/visual_transformer.pt")





def conditional_autoencoder_model(label_dim = label_dim, dropout = dropout, latent_space_size = noise_dim, expected_output_dim = expected_output_dim, batch_size = batch_size, epochs = epochs, lr = lr,
                      weight_decay = weight_decay, load_model = load_model, rotations = rotations, mask_prob = mask_prob, blank_probability = blank_probability):
    
    model = ConditionalAutoEncoder(
        n_of_classes=10, 
        dropout=dropout,
        latent_space_size=latent_space_size,
        expected_output_dim=expected_output_dim
    )

    if load_model:
        model.load_state_dict(torch.load("models/conditional_autoencoder.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform_list = [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, ), (0.5, ))
        ]
    if rotations > 0:
        transform_list.insert(0, transforms.RandomRotation(rotations))
    transform = transforms.Compose(transform_list)

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=4)
    
    '''
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    '''

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            # Apply mask to images
            mask = (torch.rand(images.size()) < mask_prob).to(device)
            blurred_images = torch.where(mask, torch.zeros_like(images), images).to(device)

            # Apply blank to images to incentivize the model to learn from the labels
            blurred_images = torch.where(torch.rand(images.size()).to(device) < blank_probability, torch.zeros_like(images).to(device), blurred_images).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            # Convert labels to one-hot encoded vectors
            labels_onehot = torch.zeros(labels.size(0), label_dim).to(device)
            labels_onehot.scatter_(1, labels.view(-1, 1), 1)

            # forward + backward + optimize
            outputs = model(labels_onehot, blurred_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    # Pass the dataset through the encoder
    latent_vectors = []
    for batch in testloader:
        batch = batch[0].to(device)
        latent_vector = model.encoder(batch)
        latent_vectors.append(latent_vector)
    latent_vectors = torch.cat(latent_vectors, dim=0)

    # Calculate the mean and standard deviation
    mean = latent_vectors.mean(dim=0)
    std = latent_vectors.std(dim=0)


    # Generate images from each category (0, 9)
                
    for i in range(10):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        with torch.no_grad():
            for category in range(10):
                # Set the category label
                labels_onehot = torch.zeros(1, label_dim).to(device)
                labels_onehot[0, category] = 1

                # Noise generation
                if i == 0:
                    noise = torch.zeros(1, latent_space_size).to(device)
                else:
                    noise = torch.normal(mean, std).to(device).unsqueeze(0)

                # Generate the output image
                output_image = model.decoder(torch.cat((noise, labels_onehot), dim=1))
                
                # Convert the output image tensor to numpy array
                output_image = output_image.squeeze().cpu().numpy()
                
                # Rescale the pixel values to [0, 1]
                output_image = (output_image + 1) / 2
                
                # Plot the generated image
                ax = axs[category // 5, category % 5]
                ax.imshow(output_image, cmap='gray')
                ax.set_title(f"Category {category}")
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'plot_autoencoder_{i}.png')

    torch.save(model.state_dict(), "models/conditional_autoencoder.pt")


    

def transformer_model(label_dim = label_dim, noise_dim = noise_dim, nhead = nhead, num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers,
                               dim_feedforward = dim_feedforward, dropout = dropout, expected_output_dim = expected_output_dim, batch_size = batch_size, epochs = epochs, lr = lr,
                               weight_decay = weight_decay, load_model = load_model, rotations = rotations):
    
    model = GenerativeImageTransformer(
        label_dim, 
        noise_dim,
        nhead, 
        num_encoder_layers, 
        num_decoder_layers, 
        dim_feedforward, 
        dropout,
        expected_output_dim=expected_output_dim
    )

    if load_model:
        model.load_state_dict(torch.load("models/generative_transformer.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform_list = [
        transforms.ToTensor(), 
        transforms.Normalize((0.5, ), (0.5, ))
        ]
    if rotations > 0:
        transform_list.insert(0, transforms.RandomRotation(rotations))
    transform = transforms.Compose(transform_list)
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=4)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                                download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=4)
    
    '''
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
    '''
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times
            
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                # Generate random noise
                current_batch_size = images.size(0)

                # Initialize an empty tensor to hold the noise vectors
                noise = torch.empty(current_batch_size, noise_dim).to(device)

                # Set the random seed for reproducibility
                for k in range(current_batch_size):
                    # Set the random seed for reproducibility
                    seed = torch.sum(images[k]*100).item()
                    torch.manual_seed(int(seed))

                    # Generate a random noise vector for this image
                    noise[k] = torch.rand(noise_dim).to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward + backward + optimize
                outputs = model(labels, noise)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
    
                # print statistics
                running_loss += loss.item()
                if i % 20 == 19:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                        (epoch + 1, i + 1, running_loss / 20))
                    running_loss = 0.0
    # Generate images from each category (0, 9)

    for i in range(10):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        with torch.no_grad():
            for category in range(10):
                # Set the category label
                label_tensor = torch.tensor([category]).to(device)

                # Generate random noise
                if i == 0:
                    noise = torch.zeros(1, noise_dim).to(device)
                else:
                    noise = torch.rand(1, noise_dim).to(device)
                
                # Generate the output image
                output_image = model(label_tensor, noise)
                
                # Convert the output image tensor to numpy array
                output_image = output_image.squeeze().cpu().numpy()
                
                # Rescale the pixel values to [0, 1]
                output_image = (output_image + 1) / 2
                
                # Plot the generated image
                ax = axs[category // 5, category % 5]
                ax.imshow(output_image, cmap='gray')
                ax.set_title(f"Category {category}")
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'plot_transformer_{i}.png')  # Save the figure to a file

    torch.save(model.state_dict(), "models/generative_transformer.pt")