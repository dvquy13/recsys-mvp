import json
import os

import lightning as L
import numpy as np
import pandas as pd
import pytest
import torch
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader

from src.features.tfm import (
    categories_pipeline_steps,
    price_pipeline_steps,
    rating_agg_pipeline_steps,
)
from src.ranker.model import Ranker

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")
CUR_DIR = os.path.abspath(os.path.join(__file__, ".."))


@pytest.fixture(scope="module")
def rank_training_data():
    user_indices = [0, 0, 1, 2, 2]
    item_indices = [0, 1, 2, 3, 4]
    timestamps = [0, 1, 2, 3, 4]
    ratings = [0, 4, 5, 3, 0]
    item_sequences = [
        [-1, -1, 2, 3],
        [-1, -1, 2, 3],
        [-1, -1, 1, 3],
        [-1, -1, 2, 1],
        [-1, -1, 2, 1],
    ]
    item_sequences_ts_buckets = [
        [-1, -1, 2, 3],
        [-1, -1, 2, 3],
        [-1, -1, 1, 3],
        [-1, -1, 2, 1],
        [-1, -1, 2, 1],
    ]
    main_category = [
        "All Electronics",
        "Video Games",
        "All Electronics",
        "Video Games",
        "Unknown",
    ]
    categories = [[], ["Headsets"], ["Video Games"], [], ["blah blah"]]
    title = [
        "World of Warcraft",
        "DotA 2",
        "Diablo IV",
        "Football Manager 2024",
        "Unknown",
    ]
    description = [[], [], ["Video games blah blah"], [], ["blah blah"]]
    price = ["from 14.99", "14.99", "price: 9.99", "20 dollars", "None"]
    parent_asin_rating_cnt_365d = [0, 1, 2, 3, 4]
    parent_asin_rating_avg_prev_rating_365d = [4.0, 3.5, 4.5, 5.0, 2.0]
    parent_asin_rating_cnt_90d = [0, 1, 2, 3, 4]
    parent_asin_rating_avg_prev_rating_90d = [4.0, 3.5, 4.5, 5.0, 2.0]
    parent_asin_rating_cnt_30d = [0, 1, 2, 3, 4]
    parent_asin_rating_avg_prev_rating_30d = [4.0, 3.5, 4.5, 5.0, 2.0]
    parent_asin_rating_cnt_7d = [0, 1, 2, 3, 4]
    parent_asin_rating_avg_prev_rating_7d = [4.0, 3.5, 4.5, 5.0, 2.0]

    train_data = {
        "user_indice": user_indices,
        "item_indice": item_indices,
        "timestamp": timestamps,
        "rating": ratings,
        "item_sequence": item_sequences,
        "item_sequence_ts_bucket": item_sequences_ts_buckets,
        "main_category": main_category,
        "title": title,
        "description": description,
        "categories": categories,
        "price": price,
        "parent_asin_rating_cnt_365d": parent_asin_rating_cnt_365d,
        "parent_asin_rating_avg_prev_rating_365d": parent_asin_rating_avg_prev_rating_365d,
        "parent_asin_rating_cnt_90d": parent_asin_rating_cnt_90d,
        "parent_asin_rating_avg_prev_rating_90d": parent_asin_rating_avg_prev_rating_90d,
        "parent_asin_rating_cnt_30d": parent_asin_rating_cnt_30d,
        "parent_asin_rating_avg_prev_rating_30d": parent_asin_rating_avg_prev_rating_30d,
        "parent_asin_rating_cnt_7d": parent_asin_rating_cnt_7d,
        "parent_asin_rating_avg_prev_rating_7d": parent_asin_rating_avg_prev_rating_7d,
    }

    return pd.DataFrame(train_data)


@pytest.fixture(scope="module")
def fit_item_metadata_pipeline(rank_training_data: pd.DataFrame):
    item_col = "item_indice"
    rating_agg_cols = [
        "parent_asin_rating_cnt_365d",
        "parent_asin_rating_avg_prev_rating_365d",
        "parent_asin_rating_cnt_90d",
        "parent_asin_rating_avg_prev_rating_90d",
        "parent_asin_rating_cnt_30d",
        "parent_asin_rating_avg_prev_rating_30d",
        "parent_asin_rating_cnt_7d",
        "parent_asin_rating_avg_prev_rating_7d",
    ]

    tfm = [
        ("main_category", OneHotEncoder(handle_unknown="ignore"), ["main_category"]),
        (
            "categories",
            Pipeline(categories_pipeline_steps()),
            "categories",
        ),  # Count Vectorizer for multi-label categorical
        (
            "price",
            Pipeline(price_pipeline_steps()),
            "price",
        ),  # Normalizing price
        (
            "rating_agg",
            Pipeline(rating_agg_pipeline_steps()),
            rating_agg_cols,
        ),
    ]

    # papermill_description=fit-tfm-pipeline
    preprocessing_pipeline = ColumnTransformer(
        transformers=tfm,
        remainder="drop",  # Drop any columns not specified in transformers
    )

    # Create a pipeline object
    item_metadata_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing_pipeline),
            (
                "normalizer",
                StandardScaler(),
            ),  # Normalize the numerical outputs since it's an important preconditions for any Deep Learning models
        ]
    )

    # Fit the pipeline
    # Drop duplicated item so that the Pipeline only fit the unique item features
    fit_df = rank_training_data.drop_duplicates(subset=[item_col])
    item_metadata_pipeline.fit(fit_df)

    return item_metadata_pipeline


def test_skipgram_forward(rank_training_data, item_metadata_pipeline):
    n_users = rank_training_data["user_indice"].nunique()
    n_items = rank_training_data["item_indice"].nunique()
    embedding_dim = 8
    batch_size = 2
    train_item_features = item_metadata_pipeline.transform(rank_training_data).astype(
        np.float32
    )

    model = Ranker(
        n_users,
        n_items,
        embedding_dim,
        item_sequence_ts_bucket_size=item_sequence_ts_bucket_size,
        bucket_embedding_dim=bucket_embedding_dim,
        item_feature_size=item_feature_size,
        dropout=dropout,
        item_embedding=item_embedding,
    )


# def test_skipgram_fit(sequences_fp: str):
#     window_size = 1
#     negative_samples = 2
#     batch_size = 2
#     n_items = 100
#     embedding_dim = 16

#     dataset = SkipGramDataset(
#         sequences_fp, window_size=window_size, negative_samples=negative_samples
#     )
#     train_loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         drop_last=False,
#         collate_fn=dataset.collate_fn,
#         num_workers=0,
#     )

#     for batch_input in train_loader:
#         print(batch_input)
#         break

#     # model
#     log_dir = f"{CUR_DIR}/logs"
#     model = SkipGram(n_items, embedding_dim)
#     lit_model = LitSkipGram(model, log_dir=log_dir)

#     # train model
#     trainer = L.Trainer(default_root_dir=f"{log_dir}/test", max_epochs=2)
#     trainer.fit(
#         model=lit_model, train_dataloaders=train_loader, val_dataloaders=train_loader
#     )
