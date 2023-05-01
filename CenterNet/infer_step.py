import torch
import cv2
import numpy as np
from CN_model import CenterNet
from config import Config
from torchvision.ops import nms

cfg = Config
weights_m = torch.load(r'.\logs_test\checkpoints\model_best2.pth')

pic_path = r'C:\Users\ITC-Admin\PycharmProjects\Detection_military\datasets\fixed_kstovo_new\images\Iter_215.png'
try:
    pic = cv2.imread(pic_path)
    assert pic is not None, "Empty image"
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
except Exception as e:
    print(f"Read image error\n{e}")
else:
    pic_tens = torch.from_numpy(pic.astype(np.float32)).permute(2, 0, 1)[None, ...]/255.0
    model = CenterNet(cfg=cfg)
    model.load_state_dict(weights_m)
    with torch.no_grad():
        out = model.inference(pic_tens, topK=50, th=0.20)

    boxes, confes, _, labels, _ = out[0]
    inds_fithered_boxes = nms(boxes, confes, iou_threshold=0.5)

    # conf_thres = 0.20
    # elems = torch.where(confes > conf_thres, confes, 0)
    # inds = elems.nonzero().squeeze(1)

    for i in inds_fithered_boxes:
        x1, y1, x2, y2 = boxes[i].to(torch.int).tolist()
        cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(pic, labels[i], (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

    # выводим изображение на экран
    cv2.imshow("Image with Target Bounding Boxes", pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

