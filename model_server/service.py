import os
import sys

import bentoml
from dotenv import load_dotenv

with bentoml.importing():
    root_dir = os.path.abspath(os.path.join(__file__, "../.."))
    sys.path.insert(0, root_dir)

load_dotenv()

model_cfg = {
    "item2vec": {"model_uri": f"models:/item2vec@champion"},
    "sequence_rating_prediction": {
        "model_uri": f"models:/sequence_rating_prediction@champion"
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


@bentoml.service(name="i2v_service")
class I2VService:
    bento_model = bentoml.models.get("item2vec")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data):
        rv = self.model.predict(input_data)
        return rv


@bentoml.service(name="seqrp_service")
class SeqRPService:
    bento_model = bentoml.models.get("sequence_rating_prediction")

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data):
        rv = self.model.predict(input_data)
        return rv
