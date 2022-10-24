import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

path_im = r"C:\Users\chere\PycharmProjects\pythonProject\Military_airplanes_detection\Data\JPEGImages\277.jpg"
image = cv2.imread(path_im)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


bbox = torch.as_tensor([[280.,  50., 393., 155.],
                        [410.,  54., 511., 150.],
                        [537.,  49., 637., 147.],
                        [155., 280., 263., 386.],
                        [283., 277., 388., 380.],
                        [408., 273., 516., 376.],
                        [541., 272., 644., 383.],
                        [417., 518., 524., 620.],
                        [415., 761., 531., 861.]
                        ], dtype=torch.float32)
labs = torch.as_tensor([14, 14, 14, 14, 14, 14, 14, 14, 14], dtype=torch.long)

transform = A.Compose([A.Resize(512, 512),
                          A.Normalize(always_apply=True,
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                          ToTensorV2()],
                          bbox_params=A.BboxParams(format='pascal_voc',
                                                  label_fields=['labels']))
sample = transform(image=image,
                   bboxes=bbox,
                    labels=labs)