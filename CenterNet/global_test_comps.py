""" Test elements of training:
* init model, model forward method and model inference
* init, compute and backward loss
* init and show dataloader objects
* train step
"""
import cv2
import torch
from config import Config
from CN_model import CenterNet
from CN_data import CenterNetDataset
from CN_loss import Loss, DIOULoss
from torch.utils.data import DataLoader

# check dataloader content
cfg = Config
try:
    dataset_obj = CenterNetDataset(classes=cfg.CLASSES_NAME, root_dir=cfg.root)
    dataload = DataLoader(dataset_obj, batch_size=cfg.batch_size, shuffle=True, collate_fn=dataset_obj.collate_fn)
except Exception as e:
    print(f"Initialization dataloader error\n{e}")

try:
    appp = iter(dataload)
    for _ in range(3):
        ll = appp.__next__()
        imgs, boxes, classes, hms, ct = ll
        for ind_in_batch in range(imgs.size(0)):
            img = imgs[ind_in_batch].permute(1, 2, 0).numpy().copy()
            for i_box in range(boxes[ind_in_batch].size(0)):
                if (boxes[ind_in_batch][i_box] == -1).sum() == 0:
                    cv2.rectangle(img,
                                  (int(boxes[ind_in_batch][i_box, 0]), int(boxes[ind_in_batch][i_box, 1])),
                                  (int(boxes[ind_in_batch][i_box, 2]), int(boxes[ind_in_batch][i_box, 3])),
                                  (0, 0, 255),
                                  1)
                    cv2.putText(img,
                                dataset_obj.id2category[classes[ind_in_batch, i_box].item()],
                                #cfg.CLASSES_NAME[classes[ind_in_batch, i_box].item()],
                                (int(boxes[ind_in_batch][i_box, 0]), int(boxes[ind_in_batch][i_box, 1])-5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                color=(0, 0, 255),
                                fontScale=0.5)
            cv2.imshow("Img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

except Exception as e:
    print(f"Error in process dataloader:\n{e}")

try:
    DEVICE = "cuda"
    model = CenterNet(cfg).to(DEVICE)
    x = [i.to(DEVICE) if isinstance(i, torch.Tensor) else i for i in ll]
    out_train = model(x[0])
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    model.cpu()
    out_inference = model.inference(x[0].cpu())
except Exception as e:
    print(f"Model init/forward/inference error\n{e}")

try:
    model.to(DEVICE)
    optimizer.zero_grad()
    loss = Loss(cfg)
    bl_ = loss(out_train, x, DEVICE)
    bl = sum(bl_)
    bl.backward()
    optimizer.step()
    out_train_after_step = model(x[0])
    print(dataset_obj.id2category)
    bl2 = sum(loss(out_train_after_step, x, DEVICE))
except Exception as e:
    print(f"Loss init/train step error:\n{e}")
else:
    print(f"Initial loss={bl}\nLoss after 1 grad step={bl2}")
    print(bl_)
