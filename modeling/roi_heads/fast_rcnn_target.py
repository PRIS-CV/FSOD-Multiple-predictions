# -*- coding: utf-8 -*-
import torch
from config import config

def bbox_transform_opr(bbox, gt):
    """ Transform the bounding box and ground truth to the loss targets.
    The 4 box coordinates are in axis 1"""
    bbox_width = bbox[:, 2] - bbox[:, 0] + 1
    bbox_height = bbox[:, 3] - bbox[:, 1] + 1
    bbox_ctr_x = bbox[:, 0] + 0.5 * bbox_width
    bbox_ctr_y = bbox[:, 1] + 0.5 * bbox_height

    gt_width = gt[:, 2] - gt[:, 0] + 1
    gt_height = gt[:, 3] - gt[:, 1] + 1
    gt_ctr_x = gt[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt[:, 1] + 0.5 * gt_height

    target_dx = (gt_ctr_x - bbox_ctr_x) / bbox_width
    target_dy = (gt_ctr_y - bbox_ctr_y) / bbox_height
    target_dw = torch.log(gt_width / bbox_width)
    target_dh = torch.log(gt_height / bbox_height)
    target = torch.cat((target_dx.reshape(-1, 1), target_dy.reshape(-1, 1),
                        target_dw.reshape(-1, 1), target_dh.reshape(-1, 1)), dim=1)
    return target

def box_overlap_ignore_opr(box, gt, ignore_label=-1):
    assert box.ndim == 2
    assert gt.ndim == 2
    assert gt.shape[-1] > 4
    area_box = (box[:, 2] - box[:, 0] + 1) * (box[:, 3] - box[:, 1] + 1)
    area_gt = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    width_height = torch.min(box[:, None, 2:], gt[:, 2:4]) - torch.max(
        box[:, None, :2], gt[:, :2])  # [N,M,2]
    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area_box[:, None] + area_gt - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device))
    ioa = torch.where(
        inter > 0,
        inter / (area_box[:, None]),
        torch.zeros(1, dtype=inter.dtype, device=inter.device))
    gt_ignore_mask = gt[:, 4].eq(ignore_label).repeat(box.shape[0], 1)
    iou *= ~gt_ignore_mask
    ioa *= gt_ignore_mask
    return iou, ioa

@torch.no_grad()
def fpn_roi_target(rpn_rois, im_info, gt_boxes, top_k=1):
    return_labels = []
    return_bbox_targets = []
    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        batch_inds = torch.ones((gt_boxes_perimg.shape[0], 1)).type_as(gt_boxes_perimg) * bid
        gt_rois = torch.cat([batch_inds, gt_boxes_perimg[:, :4]], axis=1)
        batch_roi_inds = torch.nonzero(rpn_rois[:, 0] == bid, as_tuple=False).flatten()
        all_rois = torch.cat([rpn_rois[batch_roi_inds], gt_rois], axis=0)
        overlaps_normal, overlaps_ignore = box_overlap_ignore_opr(
                all_rois[:, 1:5], gt_boxes_perimg)
        overlaps_normal, overlaps_normal_indices = overlaps_normal.sort(descending=True, dim=1)
        overlaps_ignore, overlaps_ignore_indices = overlaps_ignore.sort(descending=True, dim=1)
        # gt max and indices, ignore max and indices
        max_overlaps_normal = overlaps_normal[:, :top_k].flatten()
        gt_assignment_normal = overlaps_normal_indices[:, :top_k].flatten()
        max_overlaps_ignore = overlaps_ignore[:, :top_k].flatten()
        gt_assignment_ignore = overlaps_ignore_indices[:, :top_k].flatten()
        # cons masks
        ignore_assign_mask = (max_overlaps_normal < 0.5) * (
                max_overlaps_ignore > max_overlaps_normal)
        max_overlaps = max_overlaps_normal * ~ignore_assign_mask + \
                max_overlaps_ignore * ignore_assign_mask
        gt_assignment = gt_assignment_normal * ~ignore_assign_mask + \
                gt_assignment_ignore * ignore_assign_mask
        
        # print('gt_assignment...',gt_assignment)
        labels = gt_boxes_perimg[gt_assignment, 4]
        # print('labels...',labels)
        fg_mask = (max_overlaps >= 0.5) * (labels != -1)
        bg_mask = (max_overlaps < 0.5) * (
                max_overlaps >= 0.)
        fg_mask = fg_mask.reshape(-1, top_k)
        bg_mask = bg_mask.reshape(-1, top_k)
        pos_max = config.num_rois * config.fg_ratio
        fg_inds_mask = subsample_masks(fg_mask[:, 0], pos_max, True)
        neg_max = config.num_rois - fg_inds_mask.sum()
        bg_inds_mask = subsample_masks(bg_mask[:, 0], neg_max, True)
        labels = labels * fg_mask.flatten()
        keep_mask = fg_inds_mask + bg_inds_mask
        # labels
        labels = labels.reshape(-1, top_k)[keep_mask]
        gt_assignment = gt_assignment.reshape(-1, top_k)[keep_mask].flatten()
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        rois = all_rois[keep_mask]
        target_rois = rois.repeat(1, top_k).reshape(-1, all_rois.shape[-1])
        bbox_targets = bbox_transform_opr(target_rois[:, 1:5], target_boxes)
        if config.rcnn_bbox_normalize_targets:
            std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(bbox_targets)
            mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(bbox_targets)
            minus_opr = mean_opr / std_opr
            bbox_targets = bbox_targets / std_opr - minus_opr
        bbox_targets = bbox_targets.reshape(-1, top_k * 4)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    return_labels = torch.cat(return_labels, axis=0)
    return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
    return return_labels, return_bbox_targets

def subsample_masks(masks, num_samples, sample_value):
    positive = torch.nonzero(masks.eq(sample_value), as_tuple=False).squeeze(1)
    num_mask = len(positive)
    num_samples = int(num_samples)
    num_final_samples = min(num_mask, num_samples)
    num_final_negative = num_mask - num_final_samples
    perm = torch.randperm(num_mask, device=masks.device)[:num_final_negative]
    negative = positive[perm]
    masks[negative] = not sample_value
    return masks

