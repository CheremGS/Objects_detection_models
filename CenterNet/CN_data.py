import os
import cv2
import numpy as np
from tqdm import tqdm
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2
from CN_utils import gaussian_radius, gaussian2D, draw_umich_gaussian


class CenterNetDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform=None,
                 classes=[],
                 mode="train",
                 annots_format="pascal_voc",
                 down_stride=4):

        self.classes = classes
        self.images_dir = os.path.join(root_dir, "images")
        self.annotations_dir = os.path.join(root_dir, "annotations")
        self.transform = transform
        self.annots_format = annots_format
        self.down_stride = down_stride

        self.name_images = sorted(os.listdir(self.images_dir))
        self.name_annotations = sorted(os.listdir(self.annotations_dir))

        self.category2id = {key: value for value, key in enumerate(classes)}
        self.id2category = {key: value for value, key in self.category2id.items()}
        # check list paths
        # print(self.path_images)
        # print(self.path_annotations)

    def __len__(self):
        return len(self.name_images)

    def __getitem__(self, idx):
        # Get annotation file path for given index
        annotation_file = self.name_annotations[idx]
        image_file = self.name_images[idx]
        # Load image
        image = cv2.imread(os.path.join(self.images_dir, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        target = {"boxes": [],
                  "labels": []}

        if self.annots_format == "pascal_voc":
            # Parse XML file
            tree = ET.parse(os.path.join(self.annotations_dir, annotation_file))
            root = tree.getroot()

            # Get image path and size
            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            # Extract bounding boxes and labels from XML file
            for obj in root.findall('object'):
                label = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                target["boxes"].append([xmin, ymin, xmax, ymax])
                target["labels"].append(self.category2id[label])

        elif self.annots_format == "coco":
            with open(os.path.join(self.annotations_dir, annotation_file), 'r') as file_:
                r = file_.readlines()
                num_lists = [x.split() for x in r]
                target["labels"] += [self.category2id[x[0]] for x in num_lists]
                target["boxes"] += ([list(map(float, x[1:])) for x in num_lists])

        target['boxes'] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target["labels"], dtype=torch.long)

        # augmentations preprocessing
        if self.transform is None:
            self.transform = A.Compose([A.Normalize(max_pixel_value=255,
                                                    mean=(0, 0, 0),
                                                    std=(1, 1, 1)),
                                        ToTensorV2(p=1.0)],
                                       bbox_params={"format": "pascal_voc", "label_fields": ["labels"]})

        try:
            transformation = self.transform(image=image,
                                            bboxes=target["boxes"],
                                            labels=target["labels"])
        except Exception as e:
            print(f"Annot augment process error  in file:\n{os.path.join(self.annotations_dir, annotation_file)}")
            print(f"Error:{e}")

        bboxes_ar = np.array(transformation["bboxes"], dtype=np.int)
        boxes_w = bboxes_ar[..., 2] - bboxes_ar[..., 0]
        boxes_h = bboxes_ar[..., 3] - bboxes_ar[..., 1]
        ct = np.array([(bboxes_ar[..., 2] + bboxes_ar[..., 0])/2,
                       (bboxes_ar[..., 3] + bboxes_ar[..., 1])/2], dtype=np.float32).T

        boxes = torch.from_numpy(bboxes_ar).float()
        classes = torch.LongTensor(transformation["labels"])
        img = transformation["image"]

        outs_h, outs_w = height//self.down_stride, width//self.down_stride
        boxes_h, boxes_w, ct = boxes_h/self.down_stride, boxes_w/self.down_stride, ct/self.down_stride
        hm = np.zeros((len(self.classes), outs_h, outs_w), dtype=np.float32)
        ct[:, 0] = np.clip(ct[:, 0], 0, outs_w - 1)
        ct[:, 1] = np.clip(ct[:, 1], 0, outs_h - 1)

        obj_mask = torch.ones(len(classes))
        if os.path.basename(sys.argv[0]) == "global_test_comps.py":
            hhh = np.zeros((outs_h, outs_w))
        for i, cls_id in enumerate(classes):
            radius = gaussian_radius((np.ceil(boxes_h[i]), np.ceil(boxes_w[i])))
            radius = max(0, int(radius))
            ct_int = ct[i].astype(np.int32)
            if (hm[:, ct_int[1], ct_int[0]] == 1).sum() >= 1.:
                obj_mask[i] = 0
                continue

            draw_umich_gaussian(hm[cls_id], ct_int, radius)
            if os.path.basename(sys.argv[0]) == "global_test_comps.py":
                hhh += hm[cls_id, ...]
            if hm[cls_id, ct_int[1], ct_int[0]] != 1:
               obj_mask[i] = 0

        if os.path.basename(sys.argv[0]) == "global_test_comps.py":
            cv2.imshow("Heatmap_centers", hhh)
        hm = torch.from_numpy(hm)
        obj_mask = obj_mask.eq(1)
        boxes = boxes[obj_mask]
        classes = classes[obj_mask]

        assert hm.eq(1).sum().item() == len(classes) == len(torch.tensor(ct)[obj_mask]), \
            f"index: {idx}, hm peer: {hm.eq(1).sum().item()}, object num: {len(classes)}"
        return img, boxes, classes, hm, torch.tensor(ct)[obj_mask]

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, hm_list, ct = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []
        pad_hm_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()

        for i in range(batch_size):
            img = imgs_list[i]
            hm = hm_list[i]

            pad_imgs_list.append(torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]),
                                                               0, int(max_h - img.shape[1])), value=0.))

            pad_hm_list.append(
                torch.nn.functional.pad(hm, (0, int(max_w // 4 - hm.shape[2]),
                                             0, int(max_h // 4 - hm.shape[1])), value=0.))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)
        batch_hms = torch.stack(pad_hm_list)

        return batch_imgs, batch_boxes, batch_classes, batch_hms, ct
