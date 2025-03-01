import os

import lightning as L
import pandas as pd
import torch
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from loguru import logger
from torch import nn
from torchmetrics import AUROC, AveragePrecision

from .model import SkipGram


class LitSkipGram(L.LightningModule):
    def __init__(
        self,
        model: SkipGram,
        learning_rate: float = 0.01,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
        checkpoint_callback=None,
        accelerator: str = "cpu",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        self.accelerator = accelerator
        self.checkpoint_callback = checkpoint_callback

        # Initialize AUROC for binary classification
        self.val_roc_auc_metric = AUROC(task="binary")
        # Initialize PR-AUC for binary classification
        self.val_pr_auc_metric = AveragePrecision(task="binary")

    def training_step(self, batch, batch_idx):
        target_items = batch["target_items"]
        context_items = batch["context_items"]

        predictions = self.model.forward(target_items, context_items)
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

        predictions = self.model.forward(target_items, context_items)
        labels = batch["labels"].float().squeeze()

        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        # Update AUROC with current batch predictions and labels
        self.val_roc_auc_metric.update(predictions, labels.int())
        # Update PR-AUC with current batch predictions and labels
        self.val_pr_auc_metric.update(predictions, labels.int())

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
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

        # Compute and log the final metrics for the epoch
        roc_auc = self.val_roc_auc_metric.compute()
        pr_auc = self.val_pr_auc_metric.compute()

        self.log(
            "val_roc_auc",
            roc_auc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_pr_auc",
            pr_auc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # Reset the metrics for the next epoch
        self.val_roc_auc_metric.reset()
        self.val_pr_auc_metric.reset()

    def on_fit_end(self):
        if self.checkpoint_callback:
            logger.info(
                f"Loading best model from {self.checkpoint_callback.best_model_path}..."
            )
            self.model = LitSkipGram.load_from_checkpoint(
                self.checkpoint_callback.best_model_path, model=self.model
            ).model
        self.model = self.model.to(self._get_device())
        logger.info(f"Logging classification metrics...")
        self._log_classification_metrics()

    def _log_classification_metrics(self):
        self.model.eval()

        val_loader = self.trainer.val_dataloaders

        labels = []
        classifications = []

        for _, batch_input in enumerate(val_loader):
            _target_items = batch_input["target_items"].to(self._get_device())
            _context_items = batch_input["context_items"].to(self._get_device())
            _labels = batch_input["labels"].to(self._get_device())
            _classifications = self.model(_target_items, _context_items)

            labels.extend(_labels.cpu().detach().numpy())
            classifications.extend(_classifications.cpu().detach().numpy())

        eval_classification_df = pd.DataFrame(
            {
                "labels": labels,
                "classification_proba": classifications,
            }
        ).assign(label=lambda df: df["labels"].gt(0).astype(int))

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

            # Calculate PR-AUC using torchmetrics for MLflow
            labels_tensor = torch.tensor(
                eval_classification_df[target_col].values, dtype=torch.int
            )
            probs_tensor = torch.tensor(
                eval_classification_df["classification_proba"].values, dtype=torch.float
            )
            pr_auc_metric = AveragePrecision(task="binary")
            pr_auc = pr_auc_metric(probs_tensor, labels_tensor).item()
            mlf_client.log_metric(run_id, "pr_auc", pr_auc)

            for metric_result in classification_performance_report.as_dict()["metrics"]:
                metric = metric_result["metric"]
                if metric == "ClassificationQualityMetric":
                    roc_auc = float(metric_result["result"]["current"]["roc_auc"])
                    mlf_client.log_metric(run_id, f"roc_auc", roc_auc)
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

    def _get_device(self):
        return self.accelerator
