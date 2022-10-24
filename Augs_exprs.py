import os
path_data = r"..\Data"
print(f"\nList of data dirs {os.listdir(path_data)}")

# ---- Path dirs and amount of files
annot_dir = os.path.join(path_data, "Annotations")
annots_paths = os.listdir(annot_dir)
small_annot_dir = os.path.join(annot_dir, annots_paths[0])
images_dir = os.path.join(path_data, "JPEGImages")
TRAIN_TEST_DIVISION = os.path.join(path_data, "ImageSets", "Main")
check_dir_names = list(map(lambda x: os.path.isdir(x), [annot_dir, images_dir, TRAIN_TEST_DIVISION]))

print(f"Amount of images = {len(os.listdir(images_dir))}")
print(f"Amount of annotations = {len(os.listdir(small_annot_dir))}")


# ---- Images shapes (all pictures have different shapes)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

rand_ind = np.random.choice(np.arange(len(os.listdir(images_dir))), 10)

for ind in rand_ind:
    image_1_path = os.path.join(images_dir, os.listdir(images_dir)[ind])
    annot_1_path = os.path.join(small_annot_dir, os.listdir(small_annot_dir)[ind])
    print(os.path.isfile(annot_1_path), annot_1_path)
    # # ---- parse xml file

    labels = []
    boxes = []

    tree = ET.parse(annot_1_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        labels.append(obj.find("name").text)
        for coord in obj.findall("bndbox"):
            coords_obj = [int(coord.find(name_coord).text)
                          for name_coord in ["xmin", "ymin", "xmax", "ymax"]]
        boxes.append(coords_obj)


    # ---- drow one pic with boundbox and label
    boxes = np.array(boxes)
    image_1 = cv2.imread(image_1_path)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB).astype(np.float32)


    #image_blur = cv2.GaussianBlur(image_1, (5, 5), sigmaX=0)
    #image_blur = cv2.medianBlur(image_1, 5)

    image_canny = cv2.Canny(image_1, 100, 200)
    image_1 = cv2.resize(image_1, (512, 512))
    image_canny = cv2.resize(image_canny, (512, 512))
    thick = int((image_1.shape[0] + image_1.shape[1]) // 700)

    # for box in boxes:
    #     cv2.rectangle(image_1, (box[0], box[1]),
    #                   (box[2], box[3]), (0, 0, 0), thick)
    #
    #     cv2.putText(image_1, labels[0], (box[0], box[1] - 12),
    #                 0, 1e-3 * image_1.shape[1], (0, 0, 0), thick//3)


    cv2.imshow("Original", image_1)
    cv2.imshow("Canny", image_canny)
    # cv2.imshow("Blur", image_blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# если присмореться то заметно что в краткой аннотации рамки не попадают в границы объекта