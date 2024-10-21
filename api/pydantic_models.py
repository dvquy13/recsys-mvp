from typing import Dict, List

from pydantic import BaseModel


class FeatureRequest(BaseModel):
    entities: Dict[str, List[str]]
    features: List[str]
