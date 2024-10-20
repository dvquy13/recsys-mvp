from typing import Dict, List

from pydantic import BaseModel


class FeatureRequest(BaseModel):
    features: List[str]
    entities: Dict[str, List[str]]
