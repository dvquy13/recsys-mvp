import faiss
import numpy as np


class FaissNN:
    def __init__(self, embedding_dim, use_gpu=False, metric="L2"):
        """
        Initialize the FAISS Nearest Neighbor Search class.

        :param embedding_dim: The dimensionality of the embeddings.
        :param use_gpu: Boolean, whether to use GPU or not.
        :param metric: The distance metric to use. Options are 'L2' for Euclidean or 'IP' for Inner Product (Cosine Similarity).
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # Choose the appropriate index based on the metric
        if metric == "L2":
            self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
        elif metric == "IP":
            self.index = faiss.IndexFlatIP(
                embedding_dim
            )  # Inner Product (Cosine Similarity when embeddings are normalized)
        else:
            raise ValueError("Metric must be 'L2' or 'IP'")

        # If GPU is enabled, move the index to GPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # Initialize GPU resources
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # 0 is the GPU ID

    def add_embeddings(self, embeddings):
        """
        Add embeddings to the FAISS index.

        :param embeddings: A numpy array of embeddings, shape [n_samples, embedding_dim].
        """
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)  # Add the embeddings to the FAISS index

    def search(self, query_embedding, k=5):
        """
        Search for the k nearest neighbors of the query embedding.

        :param query_embedding: A numpy array of shape [1, embedding_dim] representing the query.
        :param k: The number of nearest neighbors to retrieve.
        :return: A tuple (distances, indices), where:
                 - distances: The distances to the k nearest neighbors.
                 - indices: The indices of the k nearest neighbors.
        """
        query_embedding = (
            np.array(query_embedding).reshape(1, -1).astype("float32")
        )  # Ensure correct shape and dtype
        distances, indices = self.index.search(query_embedding, k)  # Perform the search
        return distances, indices
