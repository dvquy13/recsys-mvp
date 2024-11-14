import json
import os

import lightning as L
import pytest
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from src.skipgram.dataset import SkipGramDataset
from src.skipgram.model import SkipGram
from src.skipgram.trainer import LitSkipGram

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")
CUR_DIR = os.path.abspath(os.path.join(__file__, ".."))


@pytest.fixture(scope="module")
def sequences_fp():
    sequences = [
        ["b", "c", "d", "e", "a"],
        ["f", "b", "k"],
        ["g", "m", "k", "l", "h"],
        ["b", "c", "k"],
        ["j", "i", "c"],
    ]

    print(f"{ROOT_DIR=}")
    print(f"{__file__=}")

    sequences_fp = f"{CUR_DIR}/sequences.jsonl"

    with open(sequences_fp, "w") as f:
        for sequence in sequences:
            f.write(json.dumps(sequence) + "\n")

    return sequences_fp


def test_skipgram_forward():
    n_items = 100
    embedding_dim = 16

    model = SkipGram(n_items, embedding_dim)

    # Example inputs
    target_items = torch.tensor([1, 2, 3])
    context_items = torch.tensor([10, 20, 30])

    # Forward pass
    predictions = model(target_items, context_items)
    assert predictions.shape == (len(target_items),)


def test_skipgram_fit(sequences_fp: str):
    window_size = 1
    negative_samples = 2
    batch_size = 2
    n_items = 100
    embedding_dim = 16

    dataset = SkipGramDataset(
        sequences_fp, window_size=window_size, negative_samples=negative_samples
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )

    for batch_input in train_loader:
        print(batch_input)
        break

    # model
    log_dir = f"{CUR_DIR}/logs"
    model = SkipGram(n_items, embedding_dim)
    lit_model = LitSkipGram(model, log_dir=log_dir)

    # train model
    trainer = L.Trainer(default_root_dir=f"{log_dir}/test", max_epochs=2)
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=train_loader
    )
