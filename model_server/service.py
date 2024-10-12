import os
import sys

import bentoml
from dotenv import load_dotenv

with bentoml.importing():
    root_dir = os.path.abspath(os.path.join(__file__, "../.."))
    sys.path.insert(0, root_dir)

load_dotenv()

model_uri = f"models:/item2vec@champion"
model_name = "item2vec"

bentoml.mlflow.import_model(
    model_name,
    model_uri=model_uri,
    signatures={
        "predict": {"batchable": True},
    },
)


@bentoml.service(name="i2v_service")
class I2VService:
    bento_model = bentoml.models.get(model_name)

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.bento_model)

    @bentoml.api
    def predict(self, input_data):
        rv = self.model.predict(input_data)
        return rv
