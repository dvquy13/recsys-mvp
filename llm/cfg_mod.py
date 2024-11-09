class LLMItemTagsCfg:
    item_tags_from_llm_fp: str = "../data/item_tags_from_llm.parquet"

    @classmethod
    def modify_run_cfg(cls, run_cfg):
        run_cfg.item_feature_cols.append("tags")
        return run_cfg
