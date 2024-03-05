import numpy as np
from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent
import torch

def cosine_similarity_representation(vectors : torch.Tensor, anchors : torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity between a batch of vectors and a set of anchor vectors.
    Args:
        vectors: The vectors to calculate the relative cosine similarity of. Expected to be a 2D tensor of shape (b, n) or a 1D tensor of shape (n).
        anchors: The set of anchor vectors to compare the vectors to. Expected to be a 2D tensor of shape (m, n).
    Returns:
        The cosine similarity between the vectors and the anchor vectors. Expected to be a 2D tensor of shape (b, m) (or (1, m) if 1D).
    """
    # If vectors is a 1D tensor (shape: (n)), unsqueeze it to make it a 2D tensor (shape: (1, n))
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(0)

    assert vectors.shape[1] == anchors.shape[1], "The vectors and the anchor vectors must have the same dimensionality."

    # Compute dot product between vectors and anchors
    dot_product = torch.matmul(vectors, anchors.t())  # shape: (b, m)

    # Compute norms of vectors and anchors
    vector_norms = torch.norm(vectors, dim=1, keepdim=True)  # shape: (b, 1)
    anchor_norms = torch.norm(anchors, dim=1, keepdim=True)  # shape: (1, m)

    # Compute cosine similarity
    cosine_similarity = dot_product / (vector_norms * anchor_norms.t())

    return cosine_similarity

def cosine_similarity_index(vector_1 : torch.Tensor, vector_2 : torch.Tensor) -> torch.Tensor:
    if len(vector_2.shape) == 1:
        return (torch.dot(vector_1, vector_2) / (torch.norm(vector_1) * torch.norm(vector_2)))
    else:
        return cosine_similarity_representation(vector_1, vector_2)

def top_k_variances(data : torch.Tensor, dimensions : int) -> torch.Tensor:
    # Calculate the variance along each dimension
    variances = torch.var(data, dim=0)

    # Get the indices of the top k variances
    _, top_k_indices = torch.topk(variances, dimensions, largest=True)

    # Select only the dimensions corresponding to the top k variances
    top_k_data = data[:, top_k_indices]

    return top_k_data

def pad_extra_dimension(np_array_1 : np.ndarray, np_array_2 : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np_array_1.shape[1] < np_array_2.shape[1]:
        np_array_1 = np.pad(np_array_1, ((0, 0), (0, np_array_2.shape[1] - np_array_1.shape[1])), mode='constant')
    elif np_array_2.shape[1] < np_array_1.shape[1]:
        np_array_2 = np.pad(np_array_2, ((0, 0), (0, np_array_1.shape[1] - np_array_2.shape[1])), mode='constant')
    return np_array_1, np_array_2

def setup_nearest_neighbours(data : np.ndarray, n_neighbours : int = 10, nndescent : bool = True) -> NNDescent:
    # Setup the nearest neighbors index
    if nndescent:
        return NNDescent(data, n_neighbors=n_neighbours)
    else:
        return NearestNeighbors(n_neighbors=n_neighbours, algorithm='ball_tree').fit(data)

def jaccard_index(s : np.ndarray, X : np.ndarray | NearestNeighbors | NNDescent, Y : np.ndarray | NearestNeighbors | NNDescent, k : int = 10) -> float:
    # Build NNDescent index for X and Y
    if isinstance(X, np.ndarray):
        X = NNDescent(X, n_neighbours=k)
    if isinstance(Y, np.ndarray):
        Y = NNDescent(Y, n_neighbours=k)

    # Ensure s is a 2D array
    if len(s.shape) == 1:
        s = s.reshape(1, -1)

    # Compute KNN in X and Y
    if isinstance(X, NNDescent):
        knn_X_indices = X.query(s, k=k)[0]
    else:
        knn_X_indices = X.kneighbors(s, n_neighbors=k, return_distance=False)

    if isinstance(Y, NNDescent):
        knn_Y_indices = Y.query(s, k=k)[0]
    else:
        knn_Y_indices = Y.kneighbors(s, n_neighbors=k, return_distance=False)

    # Compute intersection and union
    intersection = np.intersect1d(knn_X_indices, knn_Y_indices)
    union = np.union1d(knn_X_indices, knn_Y_indices)

    # Compute Jaccard index
    jaccard_index = len(intersection) / len(union)

    return jaccard_index