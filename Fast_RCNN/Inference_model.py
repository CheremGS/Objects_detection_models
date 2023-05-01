"""
Script for testing the trained fast-rcnn model.
There are two ways to check: checking for image catalogs and checking for videos:
- For the first type, you need to specify a directory with test images (only images for the test are stored in the directory)
- For the second type, you need to specify a directory with test videos (only videos for the test are in the folder)
You also need to specify the path to the stored weights of the Faster-RCNN model and txt-file with decoding data labels.
Decode file has records with string in format: "{Name_class} {label_class}"
"""

import os
import torch
import cv2
import numpy as np
from Utils import rcnn_create
from configuration import config_dataset
from Utils import plot_img_bbox

if __name__ == '__main__':
    # path to components
    dir_test_images = r"C:\Users\ITC-Admin\PycharmProjects\Detection_military\datasets\great_mavic\images"
    dir_test_videos = r"C:\Users\ITC-Admin\PycharmProjects\Detection_military\datasets\video_mavic_rescaled_1024"
    model_weight_path = r"./logs/RCNN_weights_great_mavic.pth"
    decode_txt = r"./logs/classes_great_mavic.txt"

    with open(decode_txt, "r") as decode:
        strings = decode.readlines()
    classes_dict = {int(x.split(' ')[-1].strip("\n\r")): x.split(' ')[0].strip("\n\r") for x in strings}
    list_image_names = os.listdir(dir_test_images)
    num_classes = config_dataset["num_classes"]

    if len(classes_dict) != num_classes-1:
        print("Your decode dictionary from txt-file and number of model classes are different")
        print("Recheck txt-file path and content, number of model classes")
        print(f"Amounf of decode-file classes = {len(classes_dict)}\n Amounf of classes in config = {config_dataset['num_classes']}")
        exit()

    else:
        # create model
        try:
            model = rcnn_create(num_classes-1, pretrained=True)
            model.to("cuda")

            # load weights model
            bm = torch.load(model_weight_path)
            model.load_state_dict(bm)
            model.eval()
        except Exception as e:
            print(f"Error with load model:\n{e}")
        else:
            c = 0
            while not c in [1, 2]:
                c = int(input("How to check model?\n(1 - check on dir pictures)\n(2 - check on video)\nYour choice: "))

            if c == 1:

                indexes_images = np.random.choice(len(os.listdir(dir_test_images)), 20)
                for i in indexes_images:
                    image_path = os.path.join(dir_test_images, list_image_names[i])

                    # load picture
                    image_orig = cv2.imread(image_path)
                    image = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB).astype(np.float32)
                    image = torch.from_numpy(image).permute(2, 0, 1)/255.0

                    if image_orig is None:
                        print("Recheck your path to folder with test images")
                        exit()
                    else:
                        # forward picture into model
                        with torch.no_grad():
                            out = model(image[None, ...].to("cuda"))

                        # plot image with network bboxes
                        out = out[0]
                        out = {key: out[key].cpu() for key in list(out.keys())}
                        out["boxes"] = list(out["boxes"])
                        plot_img_bbox(image.permute(1, 2, 0), out)

            elif c == 2:
                num_video = np.random.randint(low = 0, high=len(os.listdir(dir_test_videos))-1)
                cap = cv2.VideoCapture(os.path.join(dir_test_videos, os.listdir(dir_test_videos)[num_video]))
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

                while True:
                    ret, frame = cap.read()
                    if frame is None: break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
                    image = torch.from_numpy(image).permute(2, 0, 1) / 255.0

                    # forward picture into model
                    with torch.no_grad():
                        out = model(image[None, ...].to("cuda"))

                    # plot image with network bboxes
                    out = out[0]
                    out = {key: out[key].cpu() for key in list(out.keys())}
                    out["boxes"] = list(out["boxes"])

                    for ind, box in enumerate(out["boxes"]):
                        cv2.rectangle(frame,
                                      (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])),
                                      (255, 0, 0), 2)
                        cv2.putText(frame,
                                    str(classes_dict[out["labels"][ind].item()]),
                                    (int(box[0])+2, int(box[1])-3),
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale = 0.3,
                                    color = (255, 0, 0))

                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()


