import os
import sys

import bentoml
from dotenv import load_dotenv
from loguru import logger

from mlflow import MlflowClient

with bentoml.importing():
    root_dir = os.path.abspath(os.path.join(__file__, "../.."))
    sys.path.insert(0, root_dir)

load_dotenv()

model_cfg = {
    # "item2vec": {"model_uri": f"models:/item2vec@champion"},
    "sequence_rating_prediction": {
        "name": "sequence_rating_prediction",
        "deploy_alias": "champion",
        "model_uri": f"models:/sequence_rating_prediction@champion",
    },
}

for name, cfg in model_cfg.items():
    bentoml.mlflow.import_model(
        name,
        model_uri=cfg["model_uri"],
        signatures={
            "predict": {"batchable": True},
        },
    )


@bentoml.service(name="seqrp_service")
class SeqRPService:
    model_name = "sequence_rating_prediction"
    bento_model = bentoml.models.get("sequence_rating_prediction")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

        model_name = self.model_name
        deploy_alias = model_cfg.get(model_name).get("deploy_alias")

        mlf_client = MlflowClient()
        self.model_version = mlf_client.get_model_version_by_alias(
            model_name, deploy_alias
        ).version
        logger.info(
            f"Model Version for '{model_name}' with alias '{deploy_alias}': {self.model_version}"
        )

    @bentoml.api
    def predict(self, input_data):
        rv = self.model.predict(input_data)
        rv["metadata"] = {
            "model_version": self.model_version,
            "model_name": self.model_name,
        }
        return rv
