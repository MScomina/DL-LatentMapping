import torch
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from models.visual_transformer import VisualTransformer
from models.autoencoder import AutoEncoder

from utils.training_utils import get_dataset
from utils.latent_mapping_utils import cosine_similarity_representation, pad_extra_dimension, cosine_similarity_index, jaccard_index, setup_nearest_neighbours

# Parameters
n_of_anchors_per_class = 10
n_of_points = 200
alpha = 0.3
batch_size = 128
dataset_name = "cifar10"
k_neighbours = 10

def pca_comparison(n_of_anchors : int = n_of_anchors_per_class, n_of_points : int = n_of_points, alpha : float = alpha, dataset_name : str = dataset_name,
                   latent_space_size : int = 48, use_transformer : bool = False, dim_feedforward : int = 120, k_neighbours : int = k_neighbours, batch_size : int = batch_size) -> None:
    
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
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

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
            latent_space_size=latent_space_size,
            dropout=0.5,
            expected_output_dim=expected_dimension
        )

    autoencoder = AutoEncoder(
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
    visual_transformer_representations = np.empty((0, latent_space_size if not use_transformer else expected_dimension[0]*expected_dimension[1]*expected_dimension[2]))
    autoencoder_representations = np.empty((0, latent_space_size))
    visual_transformer_cosine_representations = np.empty((0, n_of_anchors*10))
    autoencoder_cosine_representations = np.empty((0, n_of_anchors*10))
    labels = np.empty((0))

    with torch.no_grad():
        visual_transformer_anchors = visual_transformer.encoder(anchors)
        if use_transformer:
            visual_transformer_anchors = visual_transformer_anchors.view(visual_transformer_anchors.size(0), -1)
        autoencoder_anchors = autoencoder.encoder(anchors)

    counter = 0
    cosine_similarity_average_normal_rep = 0
    cosine_similarity_average_cosine_rep = 0
    jaccard_average_normal_rep = 0
    jaccard_average_cosine_rep = 0

    threshold = n_of_points // 10

    for images, label in test_loader:
        images = images.to(device)
        batch_size = images.size(0)

        # Compute representations
        visual_transformer_image_representations = visual_transformer.encoder(images).view(batch_size, -1)
        autoencoder_image_representations = autoencoder.encoder(images).view(batch_size, -1)

        # Compute cosine similarities
        visual_transformer_cosine_similarities = cosine_similarity_representation(visual_transformer_image_representations, visual_transformer_anchors)
        autoencoder_cosine_similarities = cosine_similarity_representation(autoencoder_image_representations, autoencoder_anchors)

        # Compute cosine similarity indices
        if not use_transformer and autoencoder.latent_space_size == visual_transformer.latent_space_size:
            cosine_similarity_average_normal_rep += cosine_similarity_index(visual_transformer_image_representations, autoencoder_image_representations).sum().item() / batch_size
        cosine_similarity_average_cosine_rep += cosine_similarity_index(visual_transformer_cosine_similarities, autoencoder_cosine_similarities).sum().item() / batch_size


        # Append representations and labels
        visual_transformer_representations = np.concatenate((visual_transformer_representations, visual_transformer_image_representations.cpu().detach().numpy()))
        autoencoder_representations = np.concatenate((autoencoder_representations, autoencoder_image_representations.cpu().detach().numpy()))
        visual_transformer_cosine_representations = np.concatenate((visual_transformer_cosine_representations, visual_transformer_cosine_similarities.cpu().detach().numpy()))
        autoencoder_cosine_representations = np.concatenate((autoencoder_cosine_representations, autoencoder_cosine_similarities.cpu().detach().numpy()))
        labels = np.concatenate((labels, label.cpu().numpy()))

        counter += batch_size

        if counter >= threshold:
            print(f"Progress (Representations calculation): {counter}/{n_of_points}")
            threshold += n_of_points // 10
        if counter >= n_of_points:
            break


    torch.cuda.empty_cache()
    labels = labels.astype(int)

    #print(visual_transformer_representations.shape)
    #print(autoencoder_representations.shape)
    #print(visual_transformer_cosine_representations.shape)
    #print(autoencoder_cosine_representations.shape)
    #print("Visual Transformer Representations: ", visual_transformer_representations[0])
    #print("Autoencoder Representations: ", autoencoder_representations[0])
    #print("Visual Transformer Cosine Representations: ", visual_transformer_cosine_representations[0])
    #print("Autoencoder Cosine Representations: ", autoencoder_cosine_representations[0])
    X_norm = setup_nearest_neighbours(visual_transformer_representations, n_neighbours=k_neighbours, nndescent=True)
    Y_norm = setup_nearest_neighbours(autoencoder_representations, n_neighbours=k_neighbours, nndescent=True)
    X_cosine = setup_nearest_neighbours(visual_transformer_cosine_representations, n_neighbours=k_neighbours, nndescent=True)
    Y_cosine = setup_nearest_neighbours(autoencoder_cosine_representations, n_neighbours=k_neighbours, nndescent=True)
    counter = 0
    for vt_rep, ae_rep, vt_cos_rep, ae_cos_rep in zip(visual_transformer_representations, autoencoder_representations, visual_transformer_cosine_representations, autoencoder_cosine_representations):
        jaccard_average_normal_rep += jaccard_index(vt_rep, X_norm, Y_norm, k=k_neighbours)
        jaccard_average_cosine_rep += jaccard_index(vt_cos_rep, X_cosine, Y_cosine, k=k_neighbours)
        counter += 1
        if counter % (n_of_points // 10) == 0:
            print(f"Progress (Average Jaccard Similarity): {counter}/{n_of_points}")

    cosine_similarity_average_normal_rep /= (n_of_points)
    cosine_similarity_average_cosine_rep /= (n_of_points)
    jaccard_average_normal_rep /= n_of_points
    jaccard_average_cosine_rep /= n_of_points
    print("Average Cosine Similarity (Normal representation): ", cosine_similarity_average_normal_rep if not use_transformer else "N/A")
    print("Average Cosine Similarity (Cosine representation): ", cosine_similarity_average_cosine_rep)
    print("Average Jaccard Similarity (Normal representation): ", jaccard_average_normal_rep)
    print("Average Jaccard Similarity (Cosine representation): ", jaccard_average_cosine_rep)

    visual_transformer_representations, autoencoder_representations = pad_extra_dimension(visual_transformer_representations, autoencoder_representations)

    # Fit and transform the PCA on the entire dataset
    pca_1 = PCA(n_components=2)
    visual_transformer_pca_representations = pca_1.fit_transform(visual_transformer_representations)
    autoencoder_pca_representations = pca_1.transform(autoencoder_representations)

    pca_2 = PCA(n_components=2)
    autoencoder_pca_fit_representations = pca_2.fit_transform(autoencoder_representations)

    pca_3 = PCA(n_components=2)
    visual_transformer_cosine_pca_representations = pca_3.fit_transform(visual_transformer_cosine_representations)
    autoencoder_cosine_pca_representations = pca_3.transform(autoencoder_cosine_representations)

    pca_4 = PCA(n_components=2)
    autoencoder_cosine_pca_fit_representations = pca_4.fit_transform(autoencoder_cosine_representations)

    titles = [" PCA Normal Representation", " PCA Cosine Representation", " Indipendent PCA Normal Representation", " Indipendent PCA Cosine Representation"]
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
    plt.title('AutoEncoder (2) PCA Normal Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 3)
    plt.scatter(visual_transformer_cosine_pca_representations[:, 0], visual_transformer_cosine_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(titles[1])
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 4)
    plt.scatter(autoencoder_cosine_pca_representations[:, 0], autoencoder_cosine_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title('AutoEncoder (2) PCA Cosine Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    # Autoencoder
    plt.subplot(4, 2, 5)
    plt.scatter(visual_transformer_pca_representations[:, 0], visual_transformer_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(titles[2])
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 6)
    plt.scatter(autoencoder_pca_fit_representations[:, 0], autoencoder_pca_fit_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title('Autoencoder (2) Indipendent PCA Normal Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 7)
    plt.scatter(visual_transformer_cosine_pca_representations[:, 0], visual_transformer_cosine_pca_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title(titles[3])
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.subplot(4, 2, 8)
    plt.scatter(autoencoder_cosine_pca_fit_representations[:, 0], autoencoder_cosine_pca_fit_representations[:, 1], c=labels, cmap=cmap, alpha=alpha)
    plt.title('Autoencoder (2) Indipendent PCA Cosine Representation')
    plt.colorbar(ticks=range(np.min(labels), np.max(labels) + 1))

    plt.tight_layout()
    plt.show()