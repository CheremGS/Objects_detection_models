from DataGenerator import AirDataGenerator
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
    def get_train_transform():
        return A.Compose([
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0),
        ], bbox_params={
            'format': 'pascal_voc',
            'label_fields': ['labels']
        })

    # define the validation transforms
    def get_valid_transform():
        return A.Compose([A.Resize(512, 512),
                          A.Normalize(always_apply=True, mean=norm_mean, std=norm_std),
                          ToTensorV2()],
                         bbox_params=A.BboxParams(format='pascal_voc',
                                                  label_fields=['labels'])
        )

    dataset = AirDataGenerator(path_images=r"..\Data\JPEGImages",
                               path_annotations=r"..\Data\Annotations\Horizontal Bounding Boxes",
                               transform=get_valid_transform(),
                               mode="train",
                               using_canny=canny)

    num_cls = len(dataset.classes)

    indices = torch.randperm(len(dataset)).tolist()

    dataset_train = Subset(dataset, indices[0:int(0.2*len(indices))])
    dataset_test = Subset(dataset, indices[int(0.999*len(indices)):])

    dataloader_train = DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    dataloader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)

    # 2 ---- MODEL and PARAMETERS INIT SECTION ----
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
        callbacks=[dl.PeriodicLoaderCallback(valid_loader_key="valid",
                                             valid_metric_key="mAP",
                                             minimize=False,
                                             valid=1),
                   DetectionMeanAveragePrecision(num_classes=num_cls,
                                                 output_type="centernet",
                                                 iou_threshold=0.003,
                                                 confidence_threshold=0.1),
                   #dl.SchedulerCallback(metric_key="loss", loader_key="train"),
                   dl.CheckpointCallback(logdir="./logs/one",
                                         loader_key="valid",
                                         metric_key="mAP")],
        num_epochs=6,
        valid_loader="valid",
        valid_metric="mAP",
        minimize_valid_metric=False,
        verbose=True,
        loggers={"console": dl.ConsoleLogger(), "tb": dl.TensorboardLogger("./logs/tb")},
    )


