from typing import List

from loguru import logger
from pydantic import BaseModel

from llm.cfg_mod import LLMItemTagsCfg


class RunCfg(BaseModel):
    # Feature flags
    use_sbert_features: bool = True
    use_item_tags_from_llm: bool = True

    item_feature_cols: List[str] = [
        "main_category",
        "categories",
        "price",
        "parent_asin_rating_cnt_365d",
        "parent_asin_rating_avg_prev_rating_365d",
        "parent_asin_rating_cnt_90d",
        "parent_asin_rating_avg_prev_rating_90d",
        "parent_asin_rating_cnt_30d",
        "parent_asin_rating_avg_prev_rating_30d",
        "parent_asin_rating_cnt_7d",
        "parent_asin_rating_avg_prev_rating_7d",
    ]

    # Module configs
    item_tags_from_llm_fp: str = LLMItemTagsCfg.item_tags_from_llm_fp

    def init(self):
        if self.use_item_tags_from_llm:
            self = LLMItemTagsCfg.modify_run_cfg(self)

        if self.use_sbert_features:
            logger.debug(
                f"Setting use_sbert_features=True requires running notebook 016-sentence-transformers"
            )
        if self.use_item_tags_from_llm:
            logger.debug(
                f"Setting use_item_tags_from_llm=True requires running notebook 040-retrieve-item-tags-from-llm"
            )
        logger.debug(
            f"Changing use_item_tags_from_llm requires re-running notebook 002-features-v2 to get the new item_metadata_pipeline.dill file"
        )

        return self
