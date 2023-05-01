import torch
from catalyst import dl, metrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import catalyst
class CustomMAPmetric(metrics.ICallbackBatchMetric, metrics.AdditiveMetric):
    """def __init__(self, compute_on_call: bool = True):
        #super(CustomMAPmetric).__init__()
        self.compute_on_call = compute_on_call"""
    def update(self, outs: list, truth: list) -> dict:
        preds = [
            dict(
                boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0],
                                    [250.0, 40.0, 600.0, 280.0],
                                    [505.0, 105.0, 655.0, 405.0]]),
                scores=torch.tensor([0.536, 0.300, 0.600]),
                labels=torch.tensor([0, 1, 5]),
            )
        ]
        target = [
            dict(
                boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0],
                                    [500.0, 100.0, 650.0, 400.0]]),
                labels=torch.tensor([0, 3]),
            )
        ]
        mAP = MeanAveragePrecision()
        mAP.update(preds, target)
        quality = mAP.compute()
        return quality

    def update_key_value(self, *args, **kwargs) -> dict:
        pass

    def compute_key_value(self) -> dict:
        pass


class CustomRunner(dl.Runner):
    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {"loss": metrics.AdditiveMetric(compute_on_call=False)}
        if self.is_valid_loader and (self.epoch_step % 5 == 0):
            self.meters["mAP"] = metrics.AdditiveMetric(compute_on_call=False)

    def on_loader_end(self, runner):
        self.loader_metrics["loss"] = self.meters["loss"].compute()[0]
        if self.is_valid_loader and (self.epoch_step % 5 == 0):
            self.loader_metrics["mAP"] = self.meters["mAP"].compute()[0]
        super().on_loader_end(runner)

    def handle_batch(self, batch):
        x = batch[0]
        y_pred = self.model(x)
        losses = self.criterion(y_pred, batch, self.engine.device)
        sum_loss = sum(losses)

        if self.is_valid_loader and (self.epoch_step % 5 == 0):
            y_pred = self.model.inference(x, topK=50, th=0.2)

            truth = [dict(boxes=list(batch[1])[i],
                          labels=list(batch[2])[i]) for i in range(self.batch_size)]
            preds = [dict(boxes=y_pred[i][0],
                          labels=y_pred[i][2],
                          scores=y_pred[i][1]) for i in range(self.batch_size)]

            map_metric = MeanAveragePrecision(box_format="xyxy", max_detection_thresholds=[1])
            map_metric.update(preds, truth)
            map_result = map_metric.compute()
            self.batch_metrics.update({"loss": sum_loss.item(),
                                       "mAP": map_result['map']})
            self.meters["mAP"].update(self.batch_metrics["mAP"], self.batch_size)
            self.meters["loss"].update(self.batch_metrics["loss"], self.batch_size)

        else:
            self.batch_metrics.update({"Classification_loss": losses[0].item(),
                                       "Regression_loss": losses[1].item(),
                                       "loss": sum_loss.item()})
            self.meters["loss"].update(self.batch_metrics["loss"], self.batch_size)

        if self.is_train_loader:
            self.engine.backward(sum_loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
