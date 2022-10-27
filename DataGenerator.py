import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def draw_msra_gaussian(heatmap, channel, center, sigma=2):
    """Draw a gaussian on heatmap channel (inplace function).

    Args:
        heatmap (np.ndarray): heatmap matrix, expected shapes [C, W, H].
        channel (int): channel to use for drawing a gaussian.
        center (Tuple[int, int]): gaussian center coordinates.
        sigma (float): gaussian size. Default is ``2``.
    """
    tmp_size = np.float(sigma) * 6
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    _, w, h = heatmap.shape
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = (max(0, -ul[0]), min(br[0], h) - ul[0])
    g_y = (max(0, -ul[1]), min(br[1], w) - ul[1])
    img_x = (max(0, ul[0]), min(br[0], h))
    img_y = (max(0, ul[1]), min(br[1], w))
    # fmt: off
    heatmap[channel, img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[channel, img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
    )
    plt.imshow(heatmap[channel])
    plt.show()


class CenterNetDataGenerator(Dataset):
    def __init__(self,
                 path_images: str,
                 path_annotations: str,
                 transform = None,
                 using_canny = False,
                 mode="Train",
                 down_ratio = 4):

        self.mode = mode
        self.transform = transform
        self.canny = using_canny
        self.dir_images = path_images
        self.dir_annots = path_annotations
        self.down_ratio = down_ratio

        self.list_path_images = os.listdir(path_images)
        self.list_path_annots = os.listdir(path_annotations)
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

        cl = list(set(all_labels))
        with open("classes.txt", "w") as fi:
            fi.writelines("\n".join(cl) + "\n")

        return {cl[i]: i for i in range(len(cl))}

    def _get_annots_data(self, file_name):
        path_ = os.path.join(self.dir_annots, file_name)
        data_ = {"annot_path": path_,
                 "boxes": [],
                 "labels": []}

        tree = ET.parse(path_)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text

            for coord in obj.findall("bndbox"):
                coords_obj = [int(coord.find(name_coord).text)
                              for name_coord in ["xmin", "ymin", "xmax", "ymax"]]

            data_["labels"].append(self.classes[label])
            data_["boxes"].append(coords_obj)

        return data_

    def __getitem__(self, id_sample):
        # типы значений в изображении и в коробках должны быть одинаковыми (float32, а на float16 сбоит resize)!!!
        image_path = os.path.join(self.dir_images, self.list_path_images[id_sample])
        image = cv2.imread(image_path)

        if self.canny:
            image = cv2.Canny(image, 100, 200).astype(np.float32)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        target = self._get_annots_data(self.list_path_annots[id_sample])

        target["original_size"] = [image.shape[0], image.shape[1]]
        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.long)
        if self.transform:
            sample = self.transform(image=image,
                                    bboxes=target["boxes"],
                                    labels=target["labels"])
            image_transformed = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        else:
            image_transformed = torch.from_numpy((image / 255.0).astype(np.float16)).permute(2, 0, 1)

        heatmap_height = image_transformed.shape[1] // self.down_ratio
        heatmap_width = image_transformed.shape[2] // self.down_ratio
        # draw class centers
        heatmap = np.zeros(
           (self.num_classes, heatmap_height, heatmap_width), dtype=np.float32)
        for (x1, y1, x2, y2), cls_channel in zip(target['boxes'], target['labels']):
             w, h = abs(x2 - x1), abs(y2 - y1)
             xc, yc = x1 + w//2, y1 + h//2
             scaled_xc = int(xc * 1/self.down_ratio)
             scaled_yc = int(yc * 1/self.down_ratio)
             # plt.imshow(image_transformed.permute(1, 2, 0).numpy())
             # plt.show()
             # draw_msra_gaussian(heatmap, cls_channel, (scaled_xc, scaled_yc), sigma=np.clip(w * h, 2, 4))
        # draw regression squares
        wh_regr = np.zeros((2, heatmap_height, heatmap_width), dtype=np.float32)
        regrs = target['boxes'][:, 2:] - target['boxes'][:, :2]  # width, height
        for r, (x1, y1, x2, y2) in zip(regrs, target['boxes']):
             w, h = abs(x2 - x1), abs(y2 - y1)
             xc, yc = x1 + w//2, y1 + h//2
             scaled_xc = int(xc * 1/self.down_ratio)
             scaled_yc = int(yc * 1/self.down_ratio)
             for i in range(-2, 2 + 1):
                 for j in range(-2, 2 + 1):
                     try:
                         a = max(scaled_xc + i, 0)
                         b = min(scaled_yc + j, heatmap_height)
                         wh_regr[:, a, b] = r
                     except:
                         pass
        wh_regr[0] = wh_regr[0].T
        wh_regr[1] = wh_regr[1].T

        plt.imshow(wh_regr[0])
        plt.show()

        plt.imshow(wh_regr[1])
        plt.imshow()

        target["heatmap"] = torch.from_numpy(heatmap)
        target["wh_regr"] = torch.from_numpy(wh_regr)
        target['image'] = image_transformed
        target["size"] = [image_transformed.shape[1], image_transformed.shape[2]]
        return target

    def __len__(self):
        return len(self.list_path_annots)

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch, dict):
            batch = [batch]
        keys = list(batch[0].keys())
        packed_batch = {k: [] for k in keys}
        for element in batch:
            for k in keys:
                packed_batch[k].append(element[k])
        for k in ("image", "heatmap", "wh_regr"):
            packed_batch[k] = torch.stack(packed_batch[k], 0)
        return packed_batch

