import torch
from catalyst import dl


class CenterNetDetectionRunner(dl.Runner):
    """Runner for CenterNet models."""

    def handle_batch(self, batch):
        """Do a forward pass and compute loss.

        Args:
            batch (Dict[str, Any]): batch of data.
        """
        if self.is_valid_loader:
            with torch.no_grad():
                heatmaps, regression = self.model(batch["image"])
        else:
            heatmaps, regression = self.model(batch["image"])

        loss, mask_loss, regression_loss = self.criterion(
            heatmaps, regression, batch["heatmap"], batch["wh_regr"]
        )

        self.batch["predicted_heatmap"] = heatmaps
        self.batch["predicted_regression"] = regression

        # self.batch["heatmap"] = heatmaps
        # self.batch["regression"] = regression

        self.batch_metrics["mask_loss"] = mask_loss.item()
        self.batch_metrics["regression_loss"] = regression_loss.item()
        self.batch_metrics["loss"] = loss



