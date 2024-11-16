import os

import lightning as L
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

from src.dataset import UserItemBinaryDFDataset
from src.ranker.trainer import LitRanker

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")
CUR_DIR = os.path.abspath(os.path.join(__file__, ".."))


def test_ranker_fit(rank_training_data, item_metadata_pipeline, ranker_model):
    rating_col = "rating"
    timestamp_col = "timestamp"
    batch_size = 2
    k = 2
    device = "cpu"
    model = ranker_model

    user_indices = rank_training_data["user_indice"].values.tolist()
    item_indices = rank_training_data["item_indice"].values.tolist()
    item_sequences = rank_training_data["item_sequence"].values.tolist()
    item_sequences_ts_buckets = rank_training_data[
        "item_sequence_ts_bucket"
    ].values.tolist()
    train_item_features = item_metadata_pipeline.transform(rank_training_data).astype(
        np.float32
    )

    train_df = rank_training_data
    train_item_features = item_metadata_pipeline.transform(train_df).astype(np.float32)

    rating_dataset = UserItemBinaryDFDataset(
        train_df,
        "user_indice",
        "item_indice",
        rating_col,
        timestamp_col,
        item_feature=train_item_features,
    )

    train_loader = DataLoader(
        rating_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    for batch_input in train_loader:
        print(batch_input)

    # Prepare all item features for recommendation
    all_items_df = train_df.drop_duplicates(subset=["item_indice"])
    all_items_indices = all_items_df["item_indice"].values
    all_items_features = item_metadata_pipeline.transform(all_items_df).astype(
        np.float32
    )

    log_dir = f"{CUR_DIR}/logs"

    lit_model = LitRanker(
        model,
        log_dir=log_dir,
        all_items_indices=all_items_indices,
        all_items_features=all_items_features,
    )

    # Train model
    trainer = L.Trainer(
        default_root_dir=f"{log_dir}/test",
        max_epochs=200,  # Arbitrary number, if 200 epochs with this data but not able to overfit then there might be problems
        accelerator=device,
    )
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=train_loader
    )

    # Test overfit
    train_loss_epoch = trainer.callback_metrics.get("train_loss_epoch")
    logger.info(f"Latest train loss for the last epoch: {train_loss_epoch}")
    assert (
        train_loss_epoch < 0.1
    ), "Overfit 1 small batch should result in loss close to 0"

    # After fitting
    model.eval()
    users = torch.tensor(user_indices)
    items = torch.tensor(item_indices)
    item_sequences = torch.tensor(item_sequences)
    item_sequences_ts_buckets = torch.tensor(item_sequences_ts_buckets)
    item_features = torch.tensor(train_item_features)
    predictions = model.predict(
        users, item_sequences, item_sequences_ts_buckets, item_features, items
    )
    model.train()
    print(predictions)
    assert predictions.shape == (len(user_indices), 1)

    # Get the last row of each item as input for recommendations (containing the most updated item_sequence)
    to_rec_df = train_df.sort_values(timestamp_col, ascending=False).drop_duplicates(
        subset=["user_indice"]
    )
    recommendations = model.recommend(
        torch.tensor(to_rec_df["user_indice"].values.tolist()),
        torch.tensor(to_rec_df["item_sequence"].values.tolist()),
        torch.tensor(to_rec_df["item_sequence_ts_bucket"].values.tolist()),
        torch.tensor(lit_model.all_items_features),
        torch.tensor(lit_model.all_items_indices),
        k=k,
        batch_size=batch_size,
    )
    print(recommendations)
    # Expected to get top k items for each user in to_rec_df
    expected_length = to_rec_df.shape[0] * k
    assert len(recommendations["user_indice"]) == expected_length
    assert len(recommendations["recommendation"]) == expected_length
    assert len(recommendations["score"]) == expected_length
