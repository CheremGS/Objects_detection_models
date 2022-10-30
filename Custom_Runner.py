from catalyst import dl

class CenterNetDetectionRunner(dl.Runner):
    """Runner for CenterNet models."""
    def get_loaders(self, stage: str):
        loaders = super().get_loaders(stage)
        for item in loaders.values():
            if hasattr(item.dataset, "collate_fn"):
                item.collate_fn = item.dataset.collate_fn
        return loaders

    def handle_batch(self, batch):
        """Do a forward pass and compute loss.

        Args:
            batch (Dict[str, Any]): batch of data.
        """
        heatmaps, regression = self.model(batch["image"])

        loss, mask_loss, regression_loss = self.criterion(
            heatmaps, regression, batch["heatmap"], batch["wh_regr"]
        )

        self.batch["predicted_heatmap"] = heatmaps
        self.batch["predicted_regression"] = regression

        self.batch_metrics["mask_loss"] = mask_loss.item()
        self.batch_metrics["regression_loss"] = regression_loss.item()
        self.batch_metrics["loss"] = loss



