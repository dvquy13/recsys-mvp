import json
import os

import lightning as L
import pytest
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

from src.skipgram.dataset import SkipGramDataset
from src.skipgram.model import SkipGram
from src.skipgram.trainer import LitSkipGram

load_dotenv()

ROOT_DIR = os.getenv("ROOT_DIR")
CUR_DIR = os.path.abspath(os.path.join(__file__, ".."))


@pytest.fixture(scope="function")
def batch_sequences_fp():
    # This data originates from running notebook notebooks/010-prep-item2vec.ipynb.
    # But essentially just random data so can actually use the above sequences_fp.
    sequences = [
        [
            "B0006B7DXA",
            "B001LETH2Q",
            "B0009XEC02",
            "B000NNDN1M",
            "B00136MBHA",
            "B007VTVRFA",
            "B0053BCML6",
        ],
        ["B00KVP3OY8", "B07K3KHFSY", "B00KVP76G0", "B00KVOVBGM", "B0053BCML6"],
    ]

    print(f"{ROOT_DIR=}")
    print(f"{__file__=}")

    sequences_fp = f"{CUR_DIR}/batch_sequences.jsonl"

    with open(sequences_fp, "w") as f:
        for sequence in sequences:
            f.write(json.dumps(sequence) + "\n")

    return sequences_fp


def test_skipgram_overfit(batch_sequences_fp: str):
    batch_size = 1
    window_size = 1
    num_negative_samples = 2
    embedding_dim = 128

    batch_dataset = SkipGramDataset(
        batch_sequences_fp,
        window_size=window_size,
        negative_samples=num_negative_samples,
    )
    batch_train_loader = DataLoader(
        batch_dataset,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=batch_dataset.collate_fn,
        shuffle=False,
    )

    # Test data loader
    i = 0
    for batch_input in batch_train_loader:
        print(batch_input)
        i += 1
        if i >= 2:
            break

    log_dir = f"{CUR_DIR}/logs/overfit"

    # model
    model = SkipGram(len(batch_dataset.items), embedding_dim)
    lit_model = LitSkipGram(model, learning_rate=0.01, l2_reg=0.0, log_dir=log_dir)

    # train model
    trainer = L.Trainer(
        default_root_dir=log_dir,
        max_epochs=300,
        overfit_batches=1,
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=batch_train_loader,
        val_dataloaders=batch_train_loader,
    )
    logger.info(f"Logs available at {trainer.log_dir}")

    # After training is complete
    train_loss_epoch = trainer.callback_metrics.get("train_loss_epoch")
    logger.info("Latest train loss for the last epoch:", train_loss_epoch)
    assert (
        train_loss_epoch < 0.1
    ), "Overfit 1 small batch should result in loss close to 0"
