import os

import dill
import numpy as np
import pandas as pd
import torch

import mlflow
from src.ann import AnnIndex
from src.id_mapper import IDMapper


class RankerInferenceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def load_context(self, context):
        """
        This load_context method is automatically called when later we load the model.
        """
        json_path = context.artifacts["idm"]
        self.idm = IDMapper().load(json_path)

        # Qdrant
        if not (qdrant_host := os.getenv("QDRANT_HOST")):
            raise Exception(f"Environment variable QDRANT_HOST is not set.")
        qdrant_port = os.getenv("QDRANT_PORT")
        qdrant_url = f"{qdrant_host}:{qdrant_port}"
        self.ann_index = AnnIndex(
            qdrant_url=qdrant_url, qdrant_collection_name="item_desc_sbert"
        )

        item_metadata_pipeline_fp = context.artifacts["item_metadata_pipeline"]
        with open(item_metadata_pipeline_fp, "rb") as f:
            self.item_metadata_pipeline = dill.load(f)

    def predict(self, context, model_input, params=None):
        """
        Args:
            context: The context object in mlflow.pyfunc often contains pointers to artifacts that are logged alongside the model during training (like feature encoders, embeddings, etc.)
        """
        sequence_length = 10
        padding_value = -1

        if not isinstance(model_input, dict):
            # This is to work around the issue where MLflow automatically convert dict input into Dataframe
            # Ref: https://github.com/mlflow/mlflow/issues/11930
            model_input = model_input.to_dict(orient="records")[0]
        user_indices = [self.idm.get_user_index(id_) for id_ in model_input["user_ids"]]
        item_indices = [self.idm.get_item_index(id_) for id_ in model_input["item_ids"]]
        item_sequences = []
        for item_sequence in model_input["item_sequences"]:
            item_sequence = [self.idm.get_item_index(id_) for id_ in item_sequence]
            padding_needed = sequence_length - len(item_sequence)
            item_sequence = np.pad(
                item_sequence,
                (padding_needed, 0),
                "constant",
                constant_values=padding_value,
            )
            item_sequences.append(item_sequence)
        item_features = self.item_metadata_pipeline.transform(
            pd.DataFrame(model_input)
        ).astype(np.float32)
        sbert_vectors = self.ann_index.get_vector_by_ids(item_indices).astype(
            np.float32
        )
        item_features = np.hstack([item_features, sbert_vectors])

        infer_output = self.infer(
            user_indices, item_sequences, item_features, item_indices
        ).tolist()
        return {
            **model_input,
            "scores": infer_output,
        }

    def infer(self, user_indices, item_sequences, item_features, item_indices):
        user_indices = torch.tensor(user_indices)
        item_sequences = torch.tensor(item_sequences)
        item_features = torch.tensor(item_features)
        item_indices = torch.tensor(item_indices)
        output = self.model.predict(
            user_indices, item_sequences, item_features, item_indices
        )
        return output.view(len(user_indices)).detach().numpy()