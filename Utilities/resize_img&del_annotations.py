import xml.etree.ElementTree as ElemTree
import shutil
import cv2 as cv
import os

# put path (directory)
directory = 'C:\\Users\\ITC-Admin\\Desktop\\dataSet\\'
xml_folder = os.path.join(directory, 'annotations\\')
img_folder = os.path.join(directory, 'images\\')
# put path (new_folder), where you'd like to save photos
new_folder = 'C:\\Users\\ITC-Admin\\Desktop\\rescaled_detecte'
new_xml_folder = os.path.join(new_folder, 'annotations\\')
new_img_folder = os.path.join(new_folder, 'images\\')


def create_new_folders():
    os.mkdir(new_folder)
    os.mkdir(new_xml_folder)
    os.mkdir(new_img_folder)


if os.path.exists(new_folder):
    shutil.rmtree(new_folder)
    create_new_folders()
else:
    create_new_folders()
print('Введите необходимую ширину:')
w = int(input())
print('Введите необходимую высоту:')
h = int(input())

for img_path, img_dirs, img_files in os.walk(img_folder):
    for img_file in img_files:
        pic_file = os.path.join(img_path, img_file)
        img = cv.imread(pic_file)
        img_x, img_y = img.shape[1], img.shape[0]
        y1 = (img_y - h) / 2
        x1 = (img_x - w) / 2
        x2 = x1 + w
        y2 = y1 + h
        ready_img = os.path.join(new_img_folder, img_file)
        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
        cv.imwrite(ready_img, cropped_img)

for xml_path, xml_dirs, xml_files in os.walk(xml_folder):
    for xml_file in xml_files:
        ready_xml = os.path.join(new_xml_folder, xml_file)
        print(f'{ready_xml=}')
        file = os.path.join(xml_folder, xml_file)
        annot_file = open(file)
        tree = ElemTree.parse(annot_file)
        root = tree.getroot()
        list_for_remove = []
        for object in root.findall('object'):
            for xmin in object.iter('xmin'):
                new_xmin = int(float(xmin.text))
                if new_xmin < x1:
                    list_for_remove.append(object)
                else:
                    new_xmin = new_xmin - x1
                    xmin.text = str(new_xmin)
            for xmax in object.iter('xmax'):
                new_xmax = int(float(xmax.text))
                if new_xmax > x2:
                    list_for_remove.append(object)
                else:
                    new_xmax = new_xmax - x1
                    xmax.text = str(new_xmax)

        for o in list_for_remove:
            # print(f"{o=}")
            root.remove(o)

        tree.write(ready_xml)
