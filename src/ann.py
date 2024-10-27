from typing import List

import numpy as np
from qdrant_client import QdrantClient
from tqdm.auto import tqdm


class AnnIndex:
    def __init__(self, qdrant_url: str, qdrant_collection_name: str):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.qdrant_collection_name = qdrant_collection_name

        if not self.qdrant_client.collection_exists(qdrant_collection_name):
            raise Exception(
                f"Required Qdrant collection {qdrant_collection_name} does not exist"
            )

    def get_vector_by_ids(self, ids: List[int], chunk_size=100):
        records = []
        for i in tqdm(range(0, len(ids), chunk_size)):
            _ids = ids[i : i + chunk_size]
            _records = self.qdrant_client.retrieve(
                collection_name=self.qdrant_collection_name, ids=_ids, with_vectors=True
            )
            records.extend(_records)
        return np.array([record.vector for record in records])

    def get_neighbors_by_ids(self, ids: List[int], limit: int = 5):
        vector = self.get_vector_by_ids(ids)[0]
        neighbors = self.qdrant_client.search(
            collection_name=self.qdrant_collection_name,
            query_vector=vector,
            limit=limit,
        )
        return neighbors
