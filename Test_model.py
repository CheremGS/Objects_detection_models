import os
import numpy as np
import torch
from DataGenerator import CenterNetDataset
from models.centernet import CenterNet
import cv2
from Callbacks import process_centernet_output

# global params for tests
num_classes = 4
num_samples = 50 # amount of pics for model testing
model_path = "./logs/maps/best.pth" # path to checkpoints data
image_dir = "./Data/Fruits/images" # path to directory with test images

# load model from checkpoints
model = CenterNet(num_classes=num_classes)
model_weights = torch.load(model_path)['model_state_dict']
model.load_state_dict(model_weights)

# code for test forward method and criterion
# l = dataloader_train.__iter__().__next__()
# model.eval()
# X_out = model(l["image"].to(dev)) # forward method for output y_pred
# loss = criterion(X_out[0], X_out[1], l["heatmap"].to(dev), l["wh_regr"].to(dev))
# print(f'{loss=}')
# print(f'{X_out=}')
# X_out[0] and X_out[1] has shape = torch.Size([1, 20, 200, 200]), torch.Size([1, 2, 200, 200]))
# X_out[0] = classes heatmap, X_out[1] = center box heatmaps
# loss = mask_loss + regr_loss, mask_loss, regr_loss (all 1x1 tensors with grad_fn)

# load dataset for further test
with open("classes.txt", 'r') as file_classes:
    # list of labels
    list_classes = file_classes.readlines()

# list paths of image
list_images = os.listdir(image_dir)
image_paths = [list_images[i] for i in np.random.choice(a=np.arange(len(list_images)),
                                                        size=num_samples)]
# test model
model.eval()
mean_v = np.array([0.485, 0.456, 0.406])
std_v = np.array([0.229, 0.224, 0.225])
for img_path in image_paths:
    boxes = []
    image_orig = cv2.imread(os.path.join(image_dir, img_path))
    image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    image = (cv2.resize(image_orig, (512, 512)) / 255 - mean_v)/std_v

    input = torch.FloatTensor(image).permute(2, 0, 1)[None, ...]
    # freeze all gradients
    with torch.no_grad():
        outs = model(input)
        for pred_samples in process_centernet_output(outs[0], outs[1], 0, 0,
                                                     confidence_threshold=0.25,
                                                     iou_threshold=0.5,
                                                     mode="inference"):
            thick = int((image.shape[0] + image.shape[1]) // 900)
            for box in pred_samples:
                print(f"Found class {int(box[4])} with box {box[:4]}")
                thick = int((image.shape[0] + image.shape[1]) // 900)
                cv2.rectangle(image, (int(box[0]*512), int(box[1]*512)),
                              (int(box[2]*512), int(box[3]*512)), (0, 255, 0), thick)
                cv2.putText(image, list_classes[int(box[4])], (int(box[0]*512), int(box[1]*512) - 12),
                            0, 1e-3 * image.shape[1], (0, 255, 0), thick // 3)

    cv2.imshow("Resize picture", image)
    cv2.waitKey()
    cv2.destroyAllWindows()