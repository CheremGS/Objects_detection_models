import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd
import seaborn as sns


path_data = r"C:\Users\ITC-Admin\kaggle_datasets\military_airplanes"
print(f"\nList of data dirs {os.listdir(path_data)}")

# ---- Path dirs and amount of files
annot_dir = os.path.join(path_data, "Annotations")
annots_paths = os.listdir(annot_dir)
print(f"List of annotations dirs {annots_paths}")
small_annot_dir = os.path.join(annot_dir, annots_paths[0])
images_dir = os.path.join(path_data, "JPEGImages")
TRAIN_TEST_DIVISION = os.path.join(path_data, "ImageSets", "Main")
check_dir_names = list(map(lambda x: os.path.isdir(x), [annot_dir, images_dir, TRAIN_TEST_DIVISION]))
if not all(check_dir_names): print("Some dir name is wrong")

print(f"Amount of images = {len(os.listdir(images_dir))}")
print(f"Amount of annotations = {len(os.listdir(small_annot_dir))}")

# ---- Images shapes (all pictures have differents shapes)

image_1_path = os.path.join(images_dir, os.listdir(images_dir)[0])
annot_1_path = os.path.join(small_annot_dir, os.listdir(small_annot_dir)[0])
print(os.path.isfile(annot_1_path), annot_1_path)


# ---- parse xml file
tree = ET.parse(annot_1_path)
root = tree.getroot()
labels = []
boxes = []


for obj in root.findall("object"):
    labels.append(obj.find("name").text)
    coords_obj = []
    for coord in obj.findall("bndbox"):
        coords_obj = [int(coord.find(name_coord).text)
                      for name_coord in ["xmin", "ymin", "xmax", "ymax"]]
    boxes.append(coords_obj)


# ---- drow one pic with boundbox and label
boxes = np.array(boxes)
image_1 = cv2.imread(image_1_path)
thick = int((image_1.shape[0] + image_1.shape[1]) // 900)

cv2.rectangle(image_1, (boxes[0, 0], boxes[0, 1]),
              (boxes[0, 2], boxes[0, 3]), (0, 255, 0), thick)

cv2.putText(image_1, labels[0], (boxes[0, 0], boxes[0, 1] - 12),
            0, 1e-3 * image_1.shape[1], (0, 255, 0), thick//3)

cv2.imshow("Show", image_1)
cv2.waitKey()
cv2.destroyAllWindows()
# если присмореться то заметно что в краткой аннотации рамки не попадают в границы объекта

# видно, что все картинки разного размера
# аннотации ко всем написаны исходя из их каждого размера
# все приводить к одному размеру/|\обрабатывать по одной фотке в батче/|\применять что-то вроде sahi
# как-то соответственно менять аннотации/\ничего/\ничего

# сбор статистики по размеру изображений
image_shapes = np.empty((len(os.listdir(images_dir)), 3))

for i, image in enumerate(os.listdir(images_dir)):
    print(image)
    image_shapes[i] = cv2.imread(os.path.join(images_dir, image)).shape

print(image_shapes)
kwards = dict(alpha=0.5, bins=50)
plt.hist(image_shapes[:, 1], **kwards, color='orange', label='dist height')
plt.hist(image_shapes[:, 0], **kwards, color='blue', label='dist width')
plt.show()

print(f"\nHeight min-max = {image_shapes[:, 0].min()}-{image_shapes[:, 0].max()}")
print(f"Width min-max = {image_shapes[:, 1].min()}-{image_shapes[:, 1].max()}")


# check label stats
annots_list = [os.path.join(small_annot_dir, x) for x in os.listdir(small_annot_dir)]
labels = []
for annot in annots_list:
    tree = ET.parse(annot)
    root = tree.getroot()
    for obj in root.findall("object"):
         labels.append(obj.find("name").text)

pd_labs = pd.DataFrame({"labs": labels})
sns.countplot(data = pd_labs, x='labs')
plt.show()
