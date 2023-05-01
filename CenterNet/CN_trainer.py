import torch.optim
from catalyst import dl, metrics
from CN_data import CenterNetDataset
from CN_loss import Loss
from CN_model import CenterNet
from config import Config
from torch.utils.data import DataLoader
from CN_custom_runner import CustomRunner
import albumentations as A
from albumentations.pytorch import ToTensorV2

cfg = Config
dataset_obj_train = CenterNetDataset(classes=cfg.CLASSES_NAME, root_dir=cfg.root, transform=None)
dataset_obj_val = CenterNetDataset(classes=cfg.CLASSES_NAME, root_dir=cfg.root, transform=None)

loaders = {"train": DataLoader(dataset_obj_train, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset_obj_train.collate_fn),
           "valid": DataLoader(dataset_obj_val, batch_size=cfg.batch_size, collate_fn=dataset_obj_val.collate_fn)}
criterion = Loss(cfg=cfg)
model = CenterNet(cfg=cfg)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=1e-6, lr=3*1e-4, amsgrad=True)


runner = CustomRunner()
runner.train(model=model,
             optimizer=optimizer,
             loaders=loaders,
             criterion=criterion,
             scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                  factor=0.2,
                                                                  patience=4,
                                                                  threshold=1e-2,
                                                                  min_lr=1e-7,
                                                                  verbose=True),
             engine=dl.GPUEngine(),
             num_epochs=100,
             verbose=True,
             valid_loader="valid",
             valid_metric="loss",
             minimize_valid_metric=True,
             load_best_on_end=True,  # flag to load best model at the end of the training process
             logdir="./logs_draft",
             )