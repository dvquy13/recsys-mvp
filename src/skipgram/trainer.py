import os

import lightning as L
import pandas as pd
import torch
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from torch import nn

from .model import SkipGram


class LitSkipGram(L.LightningModule):
    def __init__(
        self,
        skipgram_model: SkipGram,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
    ):
        super().__init__()
        self.skipgram_model = skipgram_model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir

    def training_step(self, batch, batch_idx):
        target_items = batch["target_items"]
        context_items = batch["context_items"]

        predictions = self.skipgram_model.forward(target_items, context_items)
        labels = batch["labels"].float().squeeze()

        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        target_items = batch["target_items"]
        context_items = batch["context_items"]

        predictions = self.skipgram_model.forward(target_items, context_items)
        labels = batch["labels"].float().squeeze()

        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.Adam(
            self.skipgram_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )

        # Create the scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=2
            ),
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()

        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)

    def on_fit_end(self):
        self._log_classification_metrics(self.trainer.val_dataloaders)

    def _log_classification_metrics(self, val_loader):
        target_items = []
        context_items = []
        labels = []

        for _, batch_input in enumerate(val_loader):
            _target_items = batch_input["target_items"].cpu().detach().numpy()
            _context_items = batch_input["context_items"].cpu().detach().numpy()
            _labels = batch_input["labels"].cpu().detach().numpy()

            target_items.extend(_target_items)
            context_items.extend(_context_items)
            labels.extend(_labels)

        val_df = pd.DataFrame(
            {
                "target_items": target_items,
                "context_items": context_items,
                "labels": labels,
            }
        )

        target_items = torch.tensor(val_df["target_items"].values, device=self.device)
        context_items = torch.tensor(val_df["context_items"].values, device=self.device)
        classifications = self.skipgram_model(target_items, context_items)

        eval_classification_df = val_df.assign(
            classification_proba=classifications.cpu().detach().numpy(),
            label=lambda df: df["labels"].astype(int),
        )

        # Evidently
        target_col = "label"
        prediction_col = "classification_proba"
        column_mapping = ColumnMapping(target=target_col, prediction=prediction_col)
        classification_performance_report = Report(
            metrics=[
                ClassificationPreset(),
            ]
        )

        classification_performance_report.run(
            reference_data=None,
            current_data=eval_classification_df[[target_col, prediction_col]],
            column_mapping=column_mapping,
        )

        evidently_report_fp = f"{self.log_dir}/evidently_report_classification.html"
        os.makedirs(self.log_dir, exist_ok=True)
        classification_performance_report.save_html(evidently_report_fp)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            mlf_client = self.logger.experiment
            mlf_client.log_artifact(run_id, evidently_report_fp)
            for metric_result in classification_performance_report.as_dict()["metrics"]:
                metric = metric_result["metric"]
                if metric == "ClassificationQualityMetric":
                    roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                    mlf_client.log_metric(run_id, f"val_roc_auc", roc_auc)
                    continue
                if metric == "ClassificationPRTable":
                    columns = [
                        "top_perc",
                        "count",
                        "prob",
                        "tp",
                        "fp",
                        "precision",
                        "recall",
                    ]
                    table = metric_result["result"]["current"][1]
                    table_df = pd.DataFrame(table, columns=columns)
                    for i, row in table_df.iterrows():
                        prob = int(row["prob"] * 100)  # MLflow step only takes int
                        precision = float(row["precision"])
                        recall = float(row["recall"])
                        mlf_client.log_metric(
                            run_id,
                            f"val_precision_at_prob_as_threshold_step",
                            precision,
                            step=prob,
                        )
                        mlf_client.log_metric(
                            run_id,
                            f"val_recall_at_prob_as_threshold_step",
                            recall,
                            step=prob,
                        )
                    break
