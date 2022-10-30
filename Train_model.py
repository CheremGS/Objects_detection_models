from DataGenerator import CenterNetDataset
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from catalyst import dl
from models.centernet import CenterNet
from Criterion import CenterNetCriterion
from Callbacks import DetectionMeanAveragePrecision
from Custom_Runner import CenterNetDetectionRunner
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    canny = False
    if canny:
        norm_mean = [0.43]
        norm_std = [0.226]
    else:
        norm_mean = (0.485, 0.456, 0.406)
        norm_std = (0.229, 0.224, 0.225)

    # 1 --- DATASETS SECTION ---------
    # define transforms
    def get_transform():
        return A.Compose([
                          A.Resize(512, 512),
                          A.Normalize(always_apply=True, mean=norm_mean, std=norm_std),
                          ToTensorV2()],
                         bbox_params=A.BboxParams(format='albumentations')
        )

    dataset = CenterNetDataset(coco_json_path=r".\Data\Fruits\data.json",
                               images_dir=r".\Data\Fruits\images",
                               transforms=get_transform())

    num_cls = len(dataset.categories.keys())

    dataloader_train = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    dataloader_test = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    # 2 ---- MODEL and PARAMETERS INIT SECTION -------
    usage_cuda = True
    dev = torch.device("cuda") if (torch.cuda.is_available() and usage_cuda) else torch.device("cpu")
    print(f"{dev=}")

    if canny:
        model = CenterNet(num_classes=num_cls, input_channels=1).to(dev)
    else:
        model = CenterNet(num_classes=num_cls).to(dev)

    criterion = CenterNetCriterion(num_classes=num_cls)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001)

    loaders = {"train": dataloader_train,
               "valid": dataloader_test}

    # 3 ---- TRAIN MODEL SECTION ---------
    # catalyst
    runner = CenterNetDetectionRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        callbacks=[dl.PeriodicLoaderCallback(valid_loader_key="valid",
                                             valid_metric_key="mAP",
                                             minimize=False,
                                             valid=5),
                   DetectionMeanAveragePrecision(num_classes=num_cls,
                                                 output_type="centernet",
                                                 iou_threshold=0.7,
                                                 confidence_threshold=0.5),
                   dl.OptimizerCallback(metric_key="loss"),
                   dl.CheckpointCallback(logdir="./logs/maps",
                                         loader_key="valid",
                                         metric_key="mAP",
                                         minimize=False)],
        num_epochs=100,
        valid_loader="valid",
        valid_metric="mAP",
        minimize_valid_metric=False,
        verbose=True,
        loggers={"console": dl.ConsoleLogger(), "cs": dl.CSVLogger("./logs/cs")},
        load_best_on_end=True
    )


