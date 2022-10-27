import os
import numpy as np
import torch
from DataGenerator import CenterNetDataGenerator
from models.centernet import CenterNet
import cv2
from Callbacks import process_centernet_output

# global params for tests
num_classes = 20
num_samples = 10 # amount of pics for model testing
model_path = "./Data/last.pth" # path to checkpoints data
image_dir = "./Data/JPEGImages" # path to directory with test images

# load model from checkpoints
model = CenterNet(num_classes=num_classes, backbone="resnet34")
model_weights = torch.load(model_path)['model_state_dict']
model.load_state_dict(model_weights)

# load dataset for futher test
dataset = CenterNetDataGenerator(path_images=image_dir,
                                 path_annotations=r".\Data\Annotations\Horizontal Bounding Boxes",
                                 transform=None)
list_classes = dataset.classes # list of labels

# list paths of image
list_images = os.listdir(image_dir)
image_paths = [list_images[i] for i in np.random.choice(a=np.arange(len(list_images)),
                                                        size=num_samples)]
# test model
model.eval()
for img_path in image_paths:
    boxes = []
    image_orig = cv2.imread(os.path.join(image_dir, img_path))
    image = cv2.resize(image_orig, (512, 512))

    input = torch.FloatTensor(image).permute(2, 0, 1)[None, ...]
    # freeze all gradients
    with torch.no_grad():
        outs = model(input)
        for pred_samples in process_centernet_output(outs[0], outs[1], 0, 0,
                                                     confidence_threshold=0.1,
                                                     iou_threshold=0.1,
                                                     mode="inference"):
            thick = int((image.shape[0] + image.shape[1]) // 900)
            for box in pred_samples:
                print(f"Found class {int(box[4])} with box {box[:4]}")
                thick = int((image.shape[0] + image.shape[1]) // 900)
                cv2.rectangle(image, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 0), thick)
                cv2.putText(image, list(dataset.classes.keys())[int(box[4])], (int(box[0]), int(box[1]) - 12),
                            0, 1e-3 * image.shape[1], (0, 255, 0), thick // 3)

    cv2.imshow("Resize picture", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
