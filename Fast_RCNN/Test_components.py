"""
Testing of the basic stages of neural network training:
    * creation and preprocessing of a dataset and its visualization
    * creation of a model
    * forward-method of the model (model.predict(data))
    * one iteration of training with back-propogation
Make sure that you have correctly described the data in configuration.py
"""

import torch
import random
from DataGeneratorFCNN import RCNNDataset
from torch.utils.data import DataLoader
from Utils import rcnn_create, plot_img_bbox
from configuration import config_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

image_size = config_dataset["image_shape"]
device = "cuda"
check_all_dataset_objects = False
# create dataset
transforms_ = A.Compose([A.Resize(width=image_size[0],
                                  height=image_size[1]),
                         A.Normalize(max_pixel_value=255,
                                     mean=(0, 0, 0),
                                     std = (1, 1, 1)),
                        ToTensorV2(p=1.0)],
                       bbox_params={"format": "pascal_voc", "label_fields": ["labels"]})
dataset = RCNNDataset(path_images=config_dataset["path_image_dir"],
                      path_annotations=config_dataset["path_annots_dir"],
                      transform=transforms_)

class_labels = dataset.classes
print(f"Number of classes = {len(class_labels)}")
with open(r".\logs\classes.txt", "w") as fi:
    fi.writelines([class_label[0] + f" = {class_label[1]}\n" for class_label in class_labels.items()])

dataloader = DataLoader(dataset,
                        batch_size=config_dataset["batch_size"],
                        shuffle=False,
                        collate_fn=dataset.collate_fn)


# check generator data
try:
    tt = iter(dataloader)
    l1 = tt.__next__()
except:
    print("Dataloader create/getitem error")
else:
    print("Dataloader create/getitem is right")

# check _create all dataset objects_/_plot batch_
if check_all_dataset_objects:
    print("Check create dataset objects with annotation")
    inds = list(range(67116, len(dataset)))
    for i in tqdm(inds):
        try: img, target = dataset[i]
        except Exception as e: print(f"Error in create object #{i} - file name {dataset.list_path_annots[i]}\n{e}")
    print("Check dataset objects is completed")
else:
    inds = [random.choice(list(range(len(dataset)))) for i in range(10)]
    for i in inds:
        try:
            img, target = dataset[i]
            plot_img_bbox(img.permute(1, 2, 0), target)
        except:
            print(f"Plot batch error with object #{i}")
    print("Plotting batch successful")

# create model
try:
    model = rcnn_create(len(class_labels.items()))
    # check model forward method inference and training
    # infer
    model = model.to(device)
    model.eval()
    # images is expected to be a list of 3d tensors of shape [C, H, W]
    with torch.no_grad():
        outs = model([l1[0][0].to(device)])
    # train 2 steps
    model = rcnn_create(len(class_labels.items()))
    model.train()
    opt = torch.optim.Adam(params=model.parameters())
    outs_loss1 = model(l1[0], l1[1])
    losses1 = sum(loss for loss in outs_loss1.values())
    print(f"Losses on 1 step = {losses1}")

    losses1.backward()
    opt.step()
    opt.zero_grad()
    l2 = tt.__next__()
    outs_loss2 = model(l2[0], l2[1])
    losses2 = sum(loss for loss in outs_loss2.values())
    print(f"Losses on 2 step = {losses2}")
    assert True
except:
    print("Model_creation/forward/train-step error")
else:
    print("Model_creation/forward is right")