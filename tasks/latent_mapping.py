import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from models.visual_transformer import VisualTransformer
from models.autoencoder import AutoEncoder

from utils.training_utils import get_dataset
from utils.latent_mapping_utils import cosine_similarity_representation, top_k_variances, pad_extra_dimension

# Parameters
n_of_anchors_per_class = 10
n_of_points = 200
alpha = 0.3
dataset_name = "cifar10"

def pca_comparison(n_of_anchors : int = n_of_anchors_per_class, n_of_points : int = n_of_points, alpha : float = alpha, dataset_name : str = dataset_name,
                   latent_space_size : int = 48, use_transformer : bool = False, dim_feedforward : int = 120):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name.lower() == "mnist" or dataset_name.lower() == "fashionmnist" or dataset_name.lower() == "fmnist" or dataset_name.lower() == "kmnist":
        transform = [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, ), (0.5, ))
            ]
    else:
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

    transform = transforms.Compose(transform)

    test_set, expected_dimension = get_dataset(dataset_name, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

    if use_transformer:
        visual_transformer = VisualTransformer(
            nhead=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=dim_feedforward,
            dropout=0.5,
            patch_size=4,
            expected_dimension=expected_dimension
        )
    else:
        visual_transformer = AutoEncoder(
            linear_layers=0,
            latent_space_size=latent_space_size,
            dropout=0.5,
            expected_output_dim=expected_dimension
        )

    autoencoder = AutoEncoder(
        linear_layers=0,
        latent_space_size=latent_space_size,
        dropout=0.5,
        expected_output_dim=expected_dimension
    )


    if use_transformer:
        visual_transformer.load_state_dict(torch.load(f'models/{dataset_name}/visual_transformer - {dim_feedforward} - 0.5 - (4,3,3).pt'))
    else:
        visual_transformer.load_state_dict(torch.load(f'models/{dataset_name}/autoencoder (1) - {latent_space_size} - 0.5.pt'))
    autoencoder.load_state_dict(torch.load(f'models/{dataset_name}/autoencoder (2) - {latent_space_size} - 0.5.pt'))

    visual_transformer = visual_transformer.to(device)
    autoencoder = autoencoder.to(device)

    visual_transformer.eval()
    autoencoder.eval()

    counts = {i: 0 for i in range(10)}
    anchors = []

    for images, labels in test_set:

        if counts[labels] < n_of_anchors:
            counts[labels] += 1
            anchors.append(images)

        if all([count == n_of_anchors for count in counts.values()]):
            break

    anchors = torch.stack(anchors).to(device)

    # Initialize lists to store the representations
    visual_transformer_representations = []
    autoencoder_representations = []
    visual_transformer_cosine_representations = []
    autoencoder_cosine_representations = []
    labels = []

    with torch.no_grad():
        visual_transformer_anchors = visual_transformer.encoder(anchors)
        if use_transformer:
            visual_transformer_anchors = visual_transformer_anchors.view(visual_transformer_anchors.size(0), -1)
        autoencoder_anchors = autoencoder.encoder(anchors)

    counter = 0

    for image, label in test_loader:
        image = image.to(device)
        visual_transformer_image_representation = visual_transformer.encoder(image)
        visual_transformer_image_representation = visual_transformer_image_representation.view(-1)
        autoencoder_image_representation = autoencoder.encoder(image)
        autoencoder_image_representation = autoencoder_image_representation.view(-1)

        visual_transformer_cosine_similarity = cosine_similarity_representation(visual_transformer_image_representation, visual_transformer_anchors)
        autoencoder_cosine_similarity = cosine_similarity_representation(autoencoder_image_representation, autoencoder_anchors)

        visual_transformer_representations.append(visual_transformer_image_representation.cpu().detach().numpy().reshape(1, -1))
        autoencoder_representations.append(autoencoder_image_representation.cpu().detach().numpy().reshape(1, -1))
        visual_transformer_cosine_representations.append(visual_transformer_cosine_similarity.cpu().detach().numpy().reshape(1, -1))
        autoencoder_cosine_representations.append(autoencoder_cosine_similarity.cpu().detach().numpy().reshape(1, -1))

        labels.append(label)

        counter += 1

        if counter == n_of_points:
            break

    print("Visual Transformer Representations: ", visual_transformer_representations[0][0])
    print("Autoencoder Representations: ", autoencoder_representations[0][0])
    print("Visual Transformer Cosine Representations: ", visual_transformer_cosine_representations[0][0])
    print("Autoencoder Cosine Representations: ", autoencoder_cosine_representations[0][0])

    # Convert lists to numpy arrays
    visual_transformer_representations = np.concatenate(visual_transformer_representations)
    autoencoder_representations = np.concatenate(autoencoder_representations)
    visual_transformer_cosine_representations = np.concatenate(visual_transformer_cosine_representations)
    autoencoder_cosine_representations = np.concatenate(autoencoder_cosine_representations)
    labels = np.concatenate(labels)

    visual_transformer_representations, autoencoder_representations = pad_extra_dimension(visual_transformer_representations, autoencoder_representations)

    # Fit and transform the PCA on the entire dataset
    pca_1 = PCA(n_components=2)
    visual_transformer_pca_representations = pca_1.fit_transform(visual_transformer_representations)
    #pca_2 = PCA(n_components=2)
    #autoencoder_pca_representations = pca_2.transform(autoencoder_representations)
    autoencoder_pca_representations = pca_1.transform(autoencoder_representations)
    pca_3 = PCA(n_components=2)
    visual_transformer_cosine_pca_representations = pca_3.fit_transform(visual_transformer_cosine_representations)
    #pca_4 = PCA(n_components=2)
    #autoencoder_cosine_pca_representations = pca_4.fit_transform(autoencoder_cosine_representations)
    autoencoder_cosine_pca_representations = pca_3.transform(autoencoder_cosine_representations)

    visual_transformer_var_representations = top_k_variances(torch.tensor(visual_transformer_representations), 2).numpy()
    autoencoder_var_representations = top_k_variances(torch.tensor(autoencoder_representations), 2).numpy()
    visual_transformer_cosine_var_representations = top_k_variances(torch.tensor(visual_transformer_cosine_representations), 2).numpy()
    autoencoder_cosine_var_representations = top_k_variances(torch.tensor(autoencoder_cosine_representations), 2).numpy()

    titles = [" PCA Representation", " Cosine PCA Representation", " Top-K Var Representation", " Cosine Top-K Var Representation"]
    if use_transformer:
        titles = ["Visual Transformer" + title for title in titles]
    else:
        titles = ["AutoEncoder (1)" + title for title in titles]

    # Plot the representations
    plt.figure(figsize=(12, 12))

    cmap = plt.cm.get_cmap('viridis', np.max(labels) - np.min(labels) + 1)  # Choose a colormap

    # Visual Transformer
    plt.subplot(4, 2, 1)
    plt.scatter(visual_transformer_pca_representations[:, 0], visual_transformer_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(titles[0])
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 2)
    plt.scatter(autoencoder_pca_representations[:, 0], autoencoder_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title('AutoEncoder (2) PCA Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 3)
    plt.scatter(visual_transformer_cosine_pca_representations[:, 0], visual_transformer_cosine_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(titles[1])
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 4)
    plt.scatter(autoencoder_cosine_pca_representations[:, 0], autoencoder_cosine_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title('AutoEncoder (2) Cosine PCA Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    # Autoencoder
    plt.subplot(4, 2, 5)
    plt.scatter(visual_transformer_var_representations[:, 0], visual_transformer_var_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(titles[2])
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 6)
    plt.scatter(autoencoder_var_representations[:, 0], autoencoder_var_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title('Autoencoder (2) Top-K Var Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 7)
    plt.scatter(visual_transformer_cosine_var_representations[:, 0], visual_transformer_cosine_var_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(titles[3])
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 8)
    plt.scatter(autoencoder_cosine_var_representations[:, 0], autoencoder_cosine_var_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title('Autoencoder (2) Cosine Top-K Var Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.tight_layout()
    plt.show()
    
    '''
    # Fit and transform the t-SNE on the entire dataset
    tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
    visual_transformer_tsne_representations = tsne.fit_transform(visual_transformer_representations)
    autoencoder_tsne_representations = tsne.fit_transform(autoencoder_representations)
    visual_transformer_cosine_tsne_representations = tsne.fit_transform(visual_transformer_cosine_representations)
    autoencoder_cosine_tsne_representations = tsne.fit_transform(autoencoder_cosine_representations)

    # Plot the t-SNE representations
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(visual_transformer_tsne_representations[:, 0], visual_transformer_tsne_representations[:, 1], c=labels, cmap=cmap)
    plt.title('Visual Transformer t-SNE Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(2, 2, 2)
    plt.scatter(autoencoder_tsne_representations[:, 0], autoencoder_tsne_representations[:, 1], c=labels, cmap=cmap)
    plt.title('Autoencoder t-SNE Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(2, 2, 3)
    plt.scatter(visual_transformer_cosine_tsne_representations[:, 0], visual_transformer_cosine_tsne_representations[:, 1], c=labels, cmap=cmap)
    plt.title('Visual Transformer Cosine Similarity t-SNE Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(2, 2, 4)
    plt.scatter(autoencoder_cosine_tsne_representations[:, 0], autoencoder_cosine_tsne_representations[:, 1], c=labels, cmap=cmap)
    plt.title('Autoencoder Cosine Similarity t-SNE Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.show()
    '''