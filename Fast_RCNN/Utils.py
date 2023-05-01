"""
Utilities for creation model, visualization dataset and train/val losses
"""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from configuration import config_train, config_dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


# create model
def rcnn_create(num_classes, pretrained = True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    return model


# plot image with boundboxes
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    i = 0
    for box in (target['boxes']):
        if (box > 1024).sum() or (box < 0).sum():
            print("Frame out screen")
            i += 1
            continue
        x, y, width, height = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')
        a.text(box[0], box[1]-5, str(target['labels'][i].item()), color = "red")

        # Draw the bounding box on top of the image
        a.add_patch(rect)

        i += 1
    plt.show()


# plot training/validation loss model
def plot_losses(history):
    plt.plot(np.array(history["train"]), label="train")
    plt.plot(np.array(history["val"]), label="val")
    plt.xticks(np.arange(0, config_train["n_epochs"]), labels=np.arange(0, config_train["n_epochs"]))
    plt.grid(True)
    plt.legend()
    plt.show()
