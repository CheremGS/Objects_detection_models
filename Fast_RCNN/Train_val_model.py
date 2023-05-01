"""
Main script for training model.
Before training, it is advisable to check the correctness next parts of model fit:
    - process creation of the dataset and the model
    - structure of dataset
    - forward-method model
    - back-propogation model
    - criterion
All this can be checked using the code from Test_components.py
Make sure that you have correctly described the data in configuration.py
"""

import os
import torch
from DataGeneratorFCNN import RCNNDataset
from torch.utils.data import DataLoader, Subset
from Utils import rcnn_create, plot_losses
from configuration import config_train, config_dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

# create dataset
name_folder = os.path.basename(os.path.dirname(config_dataset["path_image_dir"]))
image_size = config_dataset["image_shape"]
transforms_ = A.Compose([A.Resize(width=image_size[0],
                                  height=image_size[1]),
                         A.Normalize(max_pixel_value=255,
                                     mean=(0, 0, 0),
                                     std = (1, 1, 1)),
                         ToTensorV2(p=1.0)
                         ],
                         bbox_params={"format": "pascal_voc",
                                      "label_fields": ["labels"]}
                        )

dataset = RCNNDataset(path_images=config_dataset["path_image_dir"],
                      path_annotations=config_dataset["path_annots_dir"],
                      transform=transforms_)

class_labels = dataset.classes
with open(f"./logs/classes_{name_folder}.txt", "w") as fi:
    fi.writelines([class_label[0] + f" = {class_label[1]}\n" for class_label in class_labels.items()])

indices = torch.randperm(len(dataset)).tolist()
dataset_train = Subset(dataset, indices[:int((1-config_dataset["test_train_proportion"])*len(indices))])
dataset_val = Subset(dataset, indices[int((1-config_dataset["test_train_proportion"])*len(indices)):])

dataloader_train = DataLoader(dataset_train,
                            shuffle=True,
                            batch_size=config_dataset["batch_size"],
                            collate_fn=dataset.collate_fn)

dataloader_val = DataLoader(dataset_val,
                            batch_size=config_dataset["batch_size"],
                            collate_fn=dataset.collate_fn)

print(f"Dataset was generated successful. \nFolders:")
print(f"Annotations - {config_dataset['path_annots_dir']}")
print(f"Images - {config_dataset['path_image_dir']}")

model = rcnn_create(len(class_labels.items()))
model = model.to(device)
optimizator = torch.optim.Adam(params=model.parameters())
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizator,
                                                          patience=config_train["patience_lr_Scheduler"],
                                                          threshold=0.0001,
                                                          min_lr=1e-6)

# variables for keeping loss history
name_losses = ["loss_classifier",
               'loss_box_reg',
                'loss_objectness',
               'loss_rpn_box_reg']
history = {"train": [], "val": []}
history_components = {"train": [], 'val': []}
best_val_loss = 1e6
previos_val_loss = 1e6
patience_for_earlystop = config_train["patience_EarlyStopping"]
threshold_for_earlystop = config_train["threshold_EarlyStopping"]
current_patience = 0

# train loop
for epoch in range(config_train["n_epochs"]):
    epoch_loss = {"train": 0, "val": 0}
    comps_losses = {"train": [0, 0, 0, 0],
                    "val": [0, 0, 0, 0]}

    with tqdm(dataloader_train, unit=" batch") as tepoch:
        for batch_ in tepoch:
            tepoch.set_description(f"Epoch {epoch} - train")

            images = tuple([img.to(device) for img in batch_[0]])
            targets = tuple({k: v.to(device) for k, v in target.items()} for target in batch_[1])

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            comps_losses["train"] = [(comps_losses["train"][i] + loss_dict[name_losses[i]].item()) for i in range(4)]
            loss_value = losses.item()
            epoch_loss["train"] += loss_value

            optimizator.zero_grad()
            losses.backward()
            optimizator.step()

            tepoch.set_postfix(loss=loss_value)

    with torch.no_grad():
        with tqdm(dataloader_val, unit=" batch") as tepoch:
            for batch_val in tepoch:
                tepoch.set_description(f"Epoch {epoch} - valid")
                images = tuple([img.to(device) for img in batch_val[0]])
                targets = tuple({k: v.to(device) for k, v in target.items()} for target in batch_val[1])

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                comps_losses["val"] = [(comps_losses["val"][i] + loss_dict[name_losses[i]].item()) for i in range(4)]
                loss_value = losses.item()
                epoch_loss["val"] += loss_value
                tepoch.set_postfix(loss=loss_value)

    train_epoch_loss = epoch_loss["train"] / len(dataloader_train)
    val_epoch_loss = epoch_loss["val"] / len(dataloader_val)
    history_components["train"].append([x/len(dataloader_train) for x in comps_losses["train"]])
    history_components["val"].append([x/len(dataloader_val) for x in comps_losses["val"]])
    history["train"].append(train_epoch_loss)
    history["val"].append(val_epoch_loss)

    print(f"Epoch {epoch} - train loss = {train_epoch_loss}")
    print(f"Epoch {epoch} - val loss = {val_epoch_loss}")

    if lr_scheduler is not None:
        curr_lr = optimizator.param_groups[0]['lr']
        lr_scheduler.step(val_epoch_loss)
        if curr_lr != optimizator.param_groups[0]['lr']:
            print(f"Current value of LEARNING RATE = {optimizator.param_groups[0]['lr']}")

    if best_val_loss > val_epoch_loss:
        best_val_loss = val_epoch_loss
        best_weights = model.state_dict()
        current_patience = 0

    if (abs(previos_val_loss-val_epoch_loss) < threshold_for_earlystop) or (-previos_val_loss+val_epoch_loss > 0):
        current_patience += 1
        print(f"Val_loss dont reduce after train on epoch #{epoch}")
    else:
        current_patience = 0

    if current_patience == patience_for_earlystop:
        print(f"EarlyStopping_Callback: stop training model on epoch #{epoch}")
        break

    previos_val_loss = val_epoch_loss

try:
    torch.save(best_weights, f"./logs/RCNN_weights_{name_folder}.pth")
except Exception as e:
    print(f"Error save model weights\n{e}")

try:
    df1 = pd.DataFrame()
    df1["train_loss"] = history["train"]
    df1["val_loss"] = history["val"]
    df2 = pd.DataFrame(history_components["train"], columns=[x + "_train" for x in name_losses])
    df3 = pd.DataFrame(history_components["val"], columns=[x + "_val" for x in name_losses])
    df = pd.concat([df1, df2, df3], axis = 1)
    df.to_excel(f'./logs/train_val_loss_{name_folder}.xlsx')
    plot_losses(history)
except Exception as e:
    print(f"Save logs error\n{e}")



