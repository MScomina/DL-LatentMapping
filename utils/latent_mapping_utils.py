import numpy as np
import torch

def cosine_similarity_representation(vector : torch.Tensor, anchors : torch.Tensor) -> torch.Tensor:
    """
    Compute the cosine similarity between a vector and a set of anchor vectors.
    Args:
        vector: The vector to calculate the relative cosine similarity of. Expected to be a 1D tensor of shape (n,).
        anchors: The set of anchor vectors to compare the vector to. Expected to be a 2D tensor of shape (m, n).
    Returns:
        The cosine similarity between the vector and the anchor vectors. Expected to be a 1D tensor of shape (m,).
    """
    assert vector.shape[0] == anchors.shape[1], "The vector and the anchor vectors must have the same dimensionality."
    return torch.matmul(vector, anchors.t()) / (torch.norm(vector) * torch.norm(anchors, dim=1)) # (vector.anchor) / (||vector|| * ||anchor||)

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