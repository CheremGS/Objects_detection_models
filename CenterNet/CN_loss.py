import torch
from torch import nn
import torch.nn.functional as F


def map2coords(h, w, stride):
    shifts_x = torch.arange(0, w * stride, step=stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, step=stride, dtype=torch.float32)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


def bbox_overlaps_diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious


def DIOULoss(pred, gt, size_num=True):
    if size_num:
        return torch.sum(1. - bbox_overlaps_diou(pred, gt)) / pred.size(0)
    return torch.sum(1. - bbox_overlaps_diou(pred, gt))


def modified_focal_loss(pred, gt):
    '''
    Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-12)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(output, mask, index, target):
    pred = gather_feature(output, index, use_transform=True)
    mask = mask.unsqueeze(dim=2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.down_stride = cfg.down_stride

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = cfg.loss_alpha
        self.beta = cfg.loss_beta
        self.gamma = cfg.loss_gamma

    def forward(self, pred, gt, dev):
        pred_hm, pred_wh, pred_offset = pred
        imgs, gt_boxes, gt_classes, gt_hm, ct = gt
        gt_nonpad_mask = gt_classes.gt(-1)

        # print('pred_hm: ', pred_hm.shape, '  gt_hm: ', gt_hm.shape)
        cls_loss = self.focal_loss(pred_hm, gt_hm)

        wh_loss = cls_loss.new_tensor(0.)
        offset_loss = cls_loss.new_tensor(0.)
        num = 0
        for batch in range(imgs.size(0)):
            # ct_batch = ct.cuda()
            ct_batch = ct[batch].to(dev)
            ct_int = ct_batch.long()
            num += len(ct_int)
            # batch_pos_pred_wh = pred_wh[batch, :, ct_int[gt_nonpad_mask[batch, :ct_int.size(0)], 1],
            #                     ct_int[gt_nonpad_mask[batch, :ct_int.size(0)], 0]].view(-1)
            batch_pos_pred_wh = pred_wh[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)
            batch_pos_pred_offset = pred_offset[batch, :, ct_int[:, 1], ct_int[:, 0]].view(-1)

            # batch_boxes = gt_boxes[batch][gt_nonpad_mask[batch]]
            batch_boxes = gt_boxes[batch][gt_nonpad_mask[batch]].to(dev)
            wh = torch.stack([
                batch_boxes[:, 2] - batch_boxes[:, 0],
                batch_boxes[:, 3] - batch_boxes[:, 1]
            ]).view(-1) / self.down_stride
            # offset = (ct - ct_int.float()).T.contiguous().view(-1)
            offset = (ct_batch - ct_int.float()).T.contiguous().view(-1)

            wh_loss += self.l1_loss(batch_pos_pred_wh, wh, reduction='sum')
            offset_loss += self.l1_loss(batch_pos_pred_offset, offset, reduction='sum')

        regr_loss = wh_loss * self.beta + offset_loss * self.gamma
        return cls_loss * self.alpha, regr_loss / (num + 1e-6)


class IOULoss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.down_stride = cfg.down_stride

        self.focal_loss = modified_focal_loss
        self.iou_loss = DIOULoss
        self.l1_loss = F.l1_loss

        self.alpha = cfg.loss_alpha
        self.beta = cfg.loss_beta
        self.gamma = cfg.loss_gamma

    def forward(self, pred, gt):
        pred_hm, pred_wh, pred_offset = pred
        imgs, gt_boxes, gt_classes, gt_hm, ct = gt
        gt_nonpad_mask = gt_classes.gt(0)

        # print('pred_hm: ', pred_hm.shape, '  gt_hm: ', gt_hm.shape)
        cls_loss = self.focal_loss(pred_hm, gt_hm)

        ### IOU LOSS ###
        output_h, output_w = pred_hm.shape[-2:]
        b, _, h, w = imgs.shape
        location = map2coords(output_h, output_w, self.down_stride).cuda()

        location = location.view(output_h, output_w, 2)
        pred_offset = pred_offset.permute(0, 2, 3, 1)
        pred_wh = pred_wh.permute(0, 2, 3, 1)
        iou_loss = cls_loss.new_tensor(0.)
        for batch in range(b):
            # ct = infos[batch]['ct']
            xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y = [[] for _ in range(6)]
            for i, cls in enumerate(gt_classes[batch][gt_nonpad_mask[batch]]):
                ct_int = ct[i]
                xs.append(location[ct_int[1], ct_int[0], 0])
                ys.append(location[ct_int[1], ct_int[0], 1])
                pos_w.append(pred_wh[batch, ct_int[1], ct_int[0], 0])
                pos_h.append(pred_wh[batch, ct_int[1], ct_int[0], 1])
                pos_offset_x.append(pred_offset[batch, ct_int[1], ct_int[0], 0])
                pos_offset_y.append(pred_offset[batch, ct_int[1], ct_int[0], 1])
            xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y = \
                [torch.stack(i) for i in [xs, ys, pos_w, pos_h, pos_offset_x, pos_offset_y]]

            det_boxes = torch.stack([
                xs - pos_w / 2 + pos_offset_x,
                ys - pos_h / 2 + pos_offset_y,
                xs + pos_w / 2 + pos_offset_x,
                ys + pos_h / 2 + pos_offset_y
            ]).T.round()

            iou_loss += self.iou_loss(det_boxes, gt_boxes[batch][gt_nonpad_mask[batch]])

        return cls_loss * self.alpha,  iou_loss / b * self.beta
