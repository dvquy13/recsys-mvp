import numpy as np
import torch


def test_ranker_forward(rank_training_data, item_metadata_pipeline, ranker_model):
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
