import numpy as np
from numpy import dtype, ndarray
from typing import Any
from sklearn import metrics


def retrieve_info(cluster_labels, y):
    """Retrieve the most common label for each cluster based on the provided labels.

    Args:
        cluster_labels (array-like): Cluster labels assigned by the KMeans algorithm.
        y (array-like): True labels corresponding to the data points.

    Returns:
        dict: A dictionary mapping cluster labels to their most common true labels.
    """
    reference_labels = {}
    for i in np.unique(cluster_labels):
        mask = cluster_labels == i
        reference_labels[i] = np.bincount(y[mask]).argmax()
    return reference_labels


def print_metrics(model, output, predicted_labels):
    """Calculate and print metrics for the KMeans clustering model.

    Args:
        model (KMeans): Trained KMeans model.
        output (array-like): True labels corresponding to the data points.
        predicted_labels (array-like): Predicted labels from the KMeans model.
    """
    print("Number of clusters: ", model.n_clusters)
    print("Inertia: ", model.inertia_)
    print("Accuracy: ", metrics.accuracy_score(predicted_labels, output))


def get_predicted_labels(model, output) -> ndarray[tuple[int], dtype[Any]]:
    """Retrieve predicted labels for each data point based on the KMeans clustering model.

    Args:
        model (KMeans): Trained KMeans model.
        output (array-like): True labels corresponding to the data points.

    Returns:
        ndarray: Array of predicted labels corresponding to the data points.
    """
    reference_labels = retrieve_info(model.labels_, output)
    predicted_labels = np.empty(len(model.labels_), dtype=int)
    for i, cluster_id in enumerate(model.labels_):
        predicted_labels[i] = reference_labels[cluster_id]
    return predicted_labels