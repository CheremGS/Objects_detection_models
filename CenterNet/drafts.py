import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
preds = [
  dict(
    # within one pic
    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0],
                        [250.0, 40.0, 600.0, 280.0],
                        [505.0, 105.0, 655.0, 405.0]]),
    scores=torch.tensor([0.536, 0.300, 0.600]),
    labels=torch.tensor([0, 1, 5]),
  ),
  dict(
    # within one pic
    boxes=torch.tensor([[258.0, 41.0, 606.0, 285.0],
                        [250.0, 40.0, 600.0, 280.0],
                        [505.0, 105.0, 655.0, 405.0]]),
    scores=torch.tensor([0.536, 0.300, 0.600]),
    labels=torch.tensor([0, 1, 3]))
]
target = [
  dict(
    # within one pic
    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0],
                        [500.0, 100.0, 650.0, 400.0]]),
    labels=torch.tensor([0, 3]),
  ),
  dict(
    # within one pic
    boxes=torch.tensor([[214.0, 41.0, 562.0, 285.0],
                        [500.0, 100.0, 650.0, 400.0]]),
    labels=torch.tensor([0, 3]),
  ),
]
metric = MeanAveragePrecision()
metric.update(preds, target)
from pprint import pprint
pprint(metric.compute())
