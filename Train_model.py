from DataGenerator import CenterNetDataGenerator
from torch.utils.data import DataLoader, Subset
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

    # 1 --- DATASETS SECTION ---- тут проблема, надо оформить разные ауги для вала и обучения но генерирует их один подкласс
    # define transforms
    def get_transform():
        return A.Compose([
                          A.Resize(512, 512),
                          A.Normalize(always_apply=True, mean=norm_mean, std=norm_std),
                          ToTensorV2()],
                         bbox_params=A.BboxParams(format='pascal_voc',
                                                  label_fields=['labels'])
        )

    dataset = CenterNetDataGenerator(path_images=r".\Data\JPEGImages",
                               path_annotations=r".\Data\Annotations\Horizontal Bounding Boxes",
                               transform=get_transform(),
                               using_canny=canny)

    num_cls = len(dataset.classes)
    indices = torch.randperm(len(dataset)).tolist()

    dataset_train = Subset(dataset, indices[0:int(0.002*len(indices))])
    dataset_test = Subset(dataset, indices[int(0.999*len(indices)):])

    dataloader_train = DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    dataloader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    # 2 ---- MODEL and PARAMETERS INIT SECTION ----
    usage_cuda = False
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

    # check forward method and criterion
    # l = dataloader_train.__iter__().__next__()
    # model.eval()
    # X_out = model(l["image"].to(dev)) # forward method for output y_pred
    # loss = criterion(X_out[0], X_out[1], l["heatmap"].to(dev), l["wh_regr"].to(dev))
    # print(f'{loss=}')
    # print(f'{X_out=}')
    # X_out[0] and X_out[1] has shape = torch.Size([1, 20, 200, 200]), torch.Size([1, 2, 200, 200]))
    # X_out[0] = classes heatmap, X_out[1] = center box heatmaps
    # loss = mask_loss + regr_loss, mask_loss, regr_loss (all 1x1 tensors with grad_fn)

    # 3 ---- TRAIN MODEL
    # catalyst
    runner = CenterNetDetectionRunner()
    runner.train(
        model=model,
        criterion=criterion,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3),
        optimizer=optimizer,
        loaders=loaders,
        callbacks=[dl.OptimizerCallback(metric_key="loss"),
                   dl.PeriodicLoaderCallback(valid_loader_key="valid",
                                             valid_metric_key="mAP",
                                             minimize=False,
                                             valid=2),
                   DetectionMeanAveragePrecision(num_classes=num_cls,
                                                 output_type="centernet",
                                                 iou_threshold=0.1,
                                                 confidence_threshold=0.1),
                   dl.CheckpointCallback(logdir="./logs/maps",
                                         loader_key="valid",
                                         metric_key="mAP")],
        num_epochs=2,
        valid_loader="valid",
        valid_metric="mAP",
        minimize_valid_metric=False,
        verbose=True,
        loggers={"console": dl.ConsoleLogger(), "cs": dl.CSVLogger("./logs/cs")},
        load_best_on_end=True
    )


