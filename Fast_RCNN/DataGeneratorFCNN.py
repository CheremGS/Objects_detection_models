import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class RCNNDataset(Dataset):
    def __init__(self,
                 path_images: str,
                 path_annotations: str,
                 transform = None):

        self.transform = transform
        self.dir_images = path_images
        self.dir_annots = path_annotations

        self.list_path_images = sorted(os.listdir(path_images))
        self.list_path_annots = sorted(os.listdir(path_annotations))
        self.classes = self.__get_classes()
        self.num_classes = len(self.classes)

    def __get_classes(self):
        list_paths_annots = [os.path.join(self.dir_annots, x) for x in self.list_path_annots]
        all_labels = []
        for file in list_paths_annots:
            root = ET.parse(file).getroot()
            labels = [obj.find("name").text for obj in root.findall("object")]
            for l in labels:
                all_labels.append(l)

        cl = sorted(list(set(all_labels)))
        return {cl[i]: i+1 for i in range(len(cl))}

    def _get_annots_data(self, file_name):
        path_ = os.path.join(self.dir_annots, file_name)
        data_ = {
                 "boxes": [],
                 "labels": []}

        tree = ET.parse(path_)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text

            for coord in obj.findall("bndbox"):
                coords_obj = [int(float(coord.find(name_coord).text))
                              for name_coord in ["xmin", "ymin", "xmax", "ymax"]]
                if all([(x > 0) and (x < 1024) for x in coords_obj]):
                    data_["labels"].append(self.classes[label])
                    data_["boxes"].append(coords_obj)

        return data_

    def __getitem__(self, id_sample):
        image_path = os.path.join(self.dir_images, self.list_path_images[id_sample])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        target = self._get_annots_data(self.list_path_annots[id_sample])

        target["image_id"] = torch.tensor([id_sample])
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        target["area"] = (target['boxes'][:, 3]-target['boxes'][:, 1])*(target['boxes'][:, 2]-target['boxes'][:, 0])
        target["iscrowd"] = torch.zeros(target["image_id"].shape[0], dtype=torch.int64)

        if self.transform:
            sample = self.transform(image=image,
                                    bboxes=target["boxes"],
                                    labels=target["labels"])
            image_transformed = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])
        else:
            image_transformed = torch.from_numpy((image / 255.0).astype(np.float32)).permute(2, 0, 1)

        return image_transformed, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.list_path_annots)