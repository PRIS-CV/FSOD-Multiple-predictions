"""Implement the CosineSimOutputLayers and  FastRCNNOutputLayers with FC layers."""
import numpy as np
import torch
from fvcore.nn import smooth_l1_loss, giou_loss
from torch import nn
from torch.nn import functional as F
from typing import Dict, Union
from torchvision.ops import box_iou

import logging
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from lvc.config import global_cfg
from detectron2.modeling.box_regression import Box2BoxTransform

ROI_HEADS_OUTPUT_REGISTRY = Registry("ROI_HEADS_OUTPUT")
ROI_HEADS_OUTPUT_REGISTRY.__doc__ = """
Registry for the output layers in ROI heads in a generalized R-CNN model."""

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
    boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(
            scores, boxes, image_shapes
        )
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero(as_tuple=False)
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
        box_reg_loss_type="smooth_l1",
        loss_weight={},
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.loss_weight = loss_weight
        self.reg_only = isinstance(self.pred_class_logits, int)
        
        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
            self.ig_reg = False
            # if proposals[0].has("gt_ignores_reg"):
            #     self.gt_ignores_reg = cat([p.gt_ignores_reg for p in proposals], dim=0)
            #     self.ig_reg = True
            # else:
            #     self.ig_reg = False
        self._no_instances = len(self.proposals) == 0
        
        self.num_classes = pred_class_logits[0].size(1) - 1 

    def _log_avg_iou(self, inference=False):
        with torch.no_grad():
            boxes = self.predict_boxes()
            boxes = torch.cat([b for b in boxes], dim=0)
            val = self.gt_classes < self.pred_class_logits
            gt_boxes = self.gt_boxes.tensor[val]
            proposal_boxes = self.proposals.tensor[val]
            boxes = boxes[val]
            gt_classes = self.gt_classes[val]
            assert gt_boxes.size() == boxes.size()
            assert proposal_boxes.size() == boxes.size()
            input_ious = torch.diag(
                box_iou(proposal_boxes, gt_boxes))
            output_ious = torch.diag(
                box_iou(boxes, gt_boxes))
            input_iou_mean = input_ious.mean().item()
            output_iou_mean = output_ious.mean().item()
            if not inference:
                storage = get_event_storage()
                storage.put_scalar(
                    "ubbr/input_iou", input_iou_mean
                )
                storage.put_scalar(
                    "ubbr/output_iou", output_iou_mean
                )
            return {
                'input_ious': input_ious, 'output_ious': output_ious,
                'gt_classes': gt_classes
            }

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits[0].argmax(dim=1)
        bg_class_ind = self.pred_class_logits[0].shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero(as_tuple=False).numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero(as_tuple=False).numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero(as_tuple=False).numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero(as_tuple=False).numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        # if self.ig:
        #     self.zero_out_logits()
        loss_cls = 0
        for pred_class_logit in self.pred_class_logits:
            loss_cls += F.cross_entropy(
                pred_class_logit, self.gt_classes, reduction="mean"
            )
        
        return loss_cls

    def zero_out_logits(self):
        num_classes = self.pred_class_logits.size(1) - 1
        shots = num_classes // 20
        z_inds = self.gt_classes != num_classes
        clses = self.gt_classes[z_inds].floor_divide(shots)
        shot_ind = self.gt_classes[z_inds].fmod(shots)
        zero_inds = \
            torch.arange(0, shots).unsqueeze(0).repeat(len(clses), 1)
        zero_inds = zero_inds.to(clses.device)
        mask = torch.ones_like(zero_inds).scatter_(1, shot_ind.unsqueeze(1), 0)
        zero_inds = zero_inds[mask.bool()].view(len(clses), -1)
        zero_inds.add_(clses.view(-1, 1)*shots)
        self.pred_class_logits[z_inds] = \
            self.pred_class_logits[z_inds].scatter(1, zero_inds, -1.e15)

    def box_reg_loss(self):
        loss_box_reg = 0
        for i in range(len(self.pred_proposal_deltas)):
            # self._log_avg_iou()
            if self._no_instances:
                return 0.0 * self.pred_proposal_deltas[i].sum()

            box_dim = self.proposals.tensor.size(1)  # 4 or 5
            cls_agnostic_bbox_reg = self.pred_proposal_deltas[i].size(1) == box_dim
            device = self.pred_proposal_deltas[i].device
            
            if not self.reg_only:
                bg_class_ind = self.pred_class_logits[i].shape[1] - 1
            else:
                bg_class_ind = self.pred_class_logits[i]
            # Box delta loss is only computed between the prediction for the gt class k
            # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
            # for non-gt classes and background.
            # Empty fg_inds should produce a valid loss of zero because reduction=sum.
            if self.ig_reg:
                fg_inds = nonzero_tuple(
                    (self.gt_classes >= 0)
                    & (self.gt_classes < bg_class_ind)
                    & (self.gt_ignores_reg == 0))[0]
            else:
                fg_inds = nonzero_tuple(
                    (self.gt_classes >= 0)
                    & (self.gt_classes < bg_class_ind))[0]

            if cls_agnostic_bbox_reg:
                # pred_proposal_deltas only corresponds to foreground class for agnostic
                gt_class_cols = torch.arange(box_dim, device=device)
            else:
                # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
                # where b is the dimension of box representation (4 or 5)
                # Note that compared to Detectron1,
                # we do not perform bounding box regression for background classes.
                gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                    box_dim, device=device
                )

            if self.box_reg_loss_type == "smooth_l1":
                gt_proposal_deltas = self.box2box_transform.get_deltas(
                    self.proposals.tensor, self.gt_boxes.tensor
                )
                loss_box_reg += smooth_l1_loss(
                    self.pred_proposal_deltas[i][fg_inds[:, None], gt_class_cols],
                    gt_proposal_deltas[fg_inds],
                    self.smooth_l1_beta,
                    reduction="sum",
                )
            elif self.box_reg_loss_type == "giou":
                fg_pred_boxes = self.box2box_transform.apply_deltas(
                    self.pred_proposal_deltas[i][fg_inds[:, None], gt_class_cols],
                    self.proposals.tensor[fg_inds],
                )
                loss_box_reg += giou_loss(
                    fg_pred_boxes,
                    self.gt_boxes.tensor[fg_inds],
                    reduction="sum",
                )
            else:
                raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def smooth_l1_loss(self, min_area=0.0):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        if not self.reg_only:
            bg_class_ind = self.pred_class_logits.shape[1] - 1
        else:
            bg_class_ind = self.pred_class_logits

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero(
            ((self.gt_classes >= 0)
             & (self.gt_classes < bg_class_ind)
             & (self.gt_boxes.area() > min_area)), as_tuple=False
        ).squeeze(1)
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def emd_softmax_loss(self, score, label, ignore_label=-1):
        with torch.no_grad():
            max_score = score.max(axis=1, keepdims=True)[0]
        score -= max_score
        log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
        mask = label != ignore_label
        vlabel = label * mask
        onehot = torch.zeros(vlabel.shape[0], self.num_classes+1, device=score.device)
        onehot.scatter_(1, vlabel.reshape(-1, 1), 1)
        loss = -(log_prob * onehot).sum(axis=1)
        loss = loss * mask
        return loss

    def emd_smooth_l1_loss(self, pred, target, beta: float):
        if beta < 1e-5:
            loss = torch.abs(pred - target.reshape(-1, 1, 4).repeat(1, self.num_classes, 1))
            loss = loss.mean(axis=1)
        else:
            abs_x = torch.abs(pred - target)
            in_mask = abs_x < beta
            loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)
        return loss.sum(axis=1)
    
    def emd_loss_unit(self, p_b0, p_s0, p_b1, p_s1, targets, labels):
        # reshape
        pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1]) #[8192,240]
        pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1]) #[8192,61]
        # print('pred_delta......',pred_delta.shape)
        # print('pred_score......',pred_score.shape)
        # p_b0 p_b1... torch.Size([2048, 240]) pred_delta... torch.Size([4096, 240])
        # p_s0 p_s1... torch.Size([2048, 61]) pred_score... torch.Size([4096, 61])
        targets = targets.reshape(-1, 4)
        labels = labels.long().flatten()
        # --------------------- Uncentain ---------------------
        targets = targets.repeat(2, 1) # torch.Size([2048, 4]) -> torch.Size([4096, 4])
        labels = labels.repeat(2,) # torch.Size([2048]) -> torch.Size([4096]) [24, 24, 24,  ..., 60, 60, 60]
        # -----------------------------------------------------
        # cons masks
        valid_masks = labels >= 0
        fg_masks = labels > 0
        # multiple class
        pred_delta = pred_delta.reshape(-1, self.num_classes, 4)
        # fg_gt_classes = labels[fg_masks] 
        # pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
        # loss for regression
        localization_loss = self.emd_smooth_l1_loss(
            pred_delta[fg_masks, :, :],
            targets[fg_masks, :],
            self.smooth_l1_beta)
        # loss for classification
        objectness_loss = self.emd_softmax_loss(pred_score, labels)
        loss = objectness_loss * valid_masks
        loss[fg_masks] = loss[fg_masks] + localization_loss
        # loss = loss.reshape(-1, 2).sum(axis=1)
        loss = loss.reshape(-1, 2).mean(axis=1)

        return loss.reshape(-1, 1)
    
    def emd_loss(self):
        loss0 = self.emd_loss_unit(
                    self.pred_proposal_deltas[0], self.pred_class_logits[0],
                    self.pred_proposal_deltas[1], self.pred_class_logits[1],
                    self.gt_boxes.tensor, self.gt_classes)
        # print('pred_proposal_deltas[0]...',self.pred_proposal_deltas[0].shape)
        # print('pred_proposal_deltas[1]...',self.pred_proposal_deltas[1].shape)
        # print('pred_class_logits[0]...',self.pred_class_logits[0].shape)
        # print('pred_class_logits[1]...',self.pred_class_logits[1].shape)
        
        loss1 = self.emd_loss_unit(
                    self.pred_proposal_deltas[1], self.pred_class_logits[1],
                    self.pred_proposal_deltas[0], self.pred_class_logits[0],
                    self.gt_boxes.tensor, self.gt_classes)
        loss = torch.cat([loss0, loss1], axis=1)
        _, min_indices = loss.min(axis=1)
        loss_emd = loss[torch.arange(loss.shape[0]), min_indices]
        loss_emd = loss_emd.mean()

        return loss_emd
    
    def diff_deltas_loss(self):
        diff_deltas_loss = 0
        k = len(self.pred_proposal_deltas)
        
        for i in range(k):
            for j in range(k):
                diff_deltas_loss -= F.l1_loss(self.pred_proposal_deltas[i], self.pred_proposal_deltas[j])
        
        return diff_deltas_loss * 0.001 / (k * k)
    
    def diff_logits_loss(self):
        diff_logits_loss = 0
        k = len(self.pred_class_logits)
        
        for i in range(k):
            for j in range(k):
                diff_logits_loss -= F.l1_loss(self.pred_class_logits[i], self.pred_class_logits[j])
        
        return diff_logits_loss * 0.001 / (k * k)
    
    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        if not self.reg_only:
            loss_dict = {"loss_cls": self.softmax_cross_entropy_loss()}
        else:
            loss_dict = {}
            self._log_avg_iou()
        loss_dict.update({"loss_box_reg": self.box_reg_loss()})
        
        # loss_dict.update({"loss_diff_deltas": self.diff_deltas_loss()})
        # loss_dict.update({"loss_diff_total": 0.3 * self.diff_logits_loss() + 0.4 * self.diff_deltas_loss()})
        
        # loss_dict = {"loss_emd": self.emd_loss()}

        return {k: v * self.loss_weight.get(k, 1.) for k, v in loss_dict.items()}

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas[0].shape[1] // B
        boxes = []
        for pred_proposal_delta in self.pred_proposal_deltas:
            boxes.append(self.box2box_transform.apply_deltas(
                pred_proposal_delta.view(num_pred * K, B),
                self.proposals.tensor.unsqueeze(1)
                .expand(num_pred, K, B)
                .reshape(-1, B),
            ))
        return torch.cat(boxes, dim=0).view(num_pred * len(self.pred_proposal_deltas), K * B).split(
            self.num_preds_per_image * len(self.pred_proposal_deltas), dim=0
        )

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        #probs = F.softmax(self.pred_class_logits, dim=-1)
        probs = F.softmax(torch.cat(self.pred_class_logits, dim = 0), dim=-1)
        #return probs.split(self.num_preds_per_image, dim=0)
        return probs.split(self.num_preds_per_image * len(self.pred_class_logits), dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """

        if not self.reg_only:
            boxes = self.predict_boxes()
            scores = self.predict_probs()
            image_shapes = self.image_shapes

            return fast_rcnn_inference(
                boxes,
                scores,
                image_shapes,
                score_thresh,
                nms_thresh,
                topk_per_image,
            )
        else:
            return self._log_avg_iou(inference=True), None
            # TODO.


@ROI_HEADS_OUTPUT_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        k = 2
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    "loss_cls" - applied to classification loss
                    "loss_box_reg" - applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        # self.fc1 = nn.Linear(input_size, 1024)
        # self.fc2 = nn.Linear(1024, 1024)

        # for l in [self.fc1, self.fc2]:
        #     nn.init.kaiming_uniform_(l.weight, a=1)
        #     nn.init.constant_(l.bias, 0)
        
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_scores = nn.ModuleList([Linear(input_size, num_classes + 1) for _ in range(k)])
        # self.cls_scores = []
        # for i in range(k):
        #     cls_score = Linear(input_size, num_classes + 1)
        #     self.add_module('cls_score{}'.format(i + 1), cls_score)
        #     self.cls_scores.append(cls_score)
        
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_preds = nn.ModuleList([Linear(input_size, num_bbox_reg_classes * box_dim) for _ in range(k)])
        # # input_size... 1024 num_classes... 60 num_bbox_reg_classes... 60 box_dim... 4
        # self.bbox_preds = []
        # for i in range(k):
        #     bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        #     self.add_module('bbox_pred{}'.format(i + 1), bbox_pred)
        #     self.bbox_preds.append(bbox_pred)

        # self.cls_score1 = Linear(input_size, num_classes + 1)
        # self.cls_score2 = Linear(input_size, num_classes + 1)
        # num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        # box_dim = len(box2box_transform.weights)
        # self.bbox_pred1 = Linear(input_size, num_bbox_reg_classes * box_dim)
        # self.bbox_pred2 = Linear(input_size, num_bbox_reg_classes * box_dim)
        

        # for l in [self.cls_score1, self.cls_score2]:
        #     nn.init.normal_(l.weight, std=0.01)
        #     nn.init.constant_(l.bias, 0)
        # for l in [self.bbox_pred1, self.bbox_pred2]:
        #     nn.init.normal_(l.weight, std=0.001)
        #     nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.k = k
        
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        # x = F.relu_(self.fc1(x))
        # x = F.relu_(self.fc2(x))

        scores, proposal_deltas = [], []
        for cls_score, bbox_pred in zip(self.cls_scores, self.bbox_preds):
            scores.append(cls_score(x))
            proposal_deltas.append(bbox_pred(x))

        # if self.k == 2:
        #     scores = [self.cls_score1(x), self.cls_score2(x)]
        #     proposal_deltas = [self.bbox_pred1(x), self.bbox_pred2(x)]
            

        return scores, proposal_deltas

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        
        losses = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            # self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


@ROI_HEADS_OUTPUT_REGISTRY.register()
class CosineSimOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        box2box_transform,
        num_classes: int,
        scale: float,
        test_score_thresh: float = 0.0,
        test_nms_thresh: float = 0.5,
        test_topk_per_image: int = 100,
        cls_agnostic_bbox_reg: bool = False,
        smooth_l1_beta: float = 0.0,
        box_reg_loss_type: str = "smooth_l1",
        loss_weight: Union[float, Dict[str, float]] = 1.0,
        k = 2
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    "loss_cls" - applied to classification loss
                    "loss_box_reg" - applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_scores = nn.ModuleList([Linear(input_size, num_classes + 1, bias=False) for _ in range(k)])
        # self.cls_scores = []
        # for i in range(k):
        #     cls_score = Linear(input_size, num_classes + 1, bias=False)
        #     self.add_module('cls_score{}'.format(i + 1), cls_score)
        #     self.cls_scores.append(cls_score)
        
        # self.scale = scale
        # if self.scale == -1:
        #     # learnable global scaling factor
        #     self.scale = nn.Parameter(torch.ones(1) * 20.0)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_preds = nn.ModuleList([Linear(input_size, num_bbox_reg_classes * box_dim) for _ in range(k)])
        # self.bbox_preds = []
        # for i in range(k):
        #     bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        #     self.add_module('bbox_pred{}'.format(i + 1), bbox_pred)
        #     self.bbox_preds.append(bbox_pred)
            
        for l in self.cls_scores:
            nn.init.normal_(l.weight, std=0.01)
        for l in self.bbox_preds:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

        # self.cls_score1 = Linear(input_size, num_classes + 1)
        # self.cls_score2 = Linear(input_size, num_classes + 1)
        # num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        # box_dim = len(box2box_transform.weights)
        # self.bbox_pred1 = Linear(input_size, num_bbox_reg_classes * box_dim)
        # self.bbox_ = Linear(input_size, num_bbox_reg_classes * box_dim)

        # for l in [self.cls_score1, self.cls_score2]:
        #     nn.init.normal_(l.weight, std=0.01)
        #     nn.init.constant_(l.bias, 0)
        # for l in [self.bbox_pred1, self.bbox_pred2]:
        #     nn.init.normal_(l.weight, std=0.001)
        #     nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.k = k

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "scale"                 : cfg.MODEL.ROI_HEADS.COSINE_SCALE,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight"           : {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)

        scores, proposal_deltas = [], []
        
        # normalize weight
        for cls_score, bbox_pred in zip(self.cls_scores, self.bbox_preds):
            temp_norm = (
                torch.norm(cls_score.weight.data, p=2, dim=1)
                .unsqueeze(1)
                .expand_as(cls_score.weight.data)
            )
            cls_score.weight.data = cls_score.weight.data.div(
                temp_norm + 1e-5
            )
            cos_dist = cls_score(x_normalized)
            scores.append(self.scale * cos_dist)
            proposal_deltas.append(bbox_pred(x))

        # if self.k == 2:
        #     temp_norm = (
        #         torch.norm(self.cls_score1.weight.data, p=2, dim=1)
        #         .unsqueeze(1)
        #         .expand_as(self.cls_score1.weight.data)
        #     )
        #     self.cls_score1.weight.data = self.cls_score1.weight.data.div(
        #         temp_norm + 1e-5
        #     )
        #     cos_dist = self.cls_score1(x_normalized)
        #     scores.append(self.scale * cos_dist)
        #     proposal_deltas.append(self.bbox_pred1(x))

        #     temp_norm = (
        #         torch.norm(self.cls_score2.weight.data, p=2, dim=1)
        #         .unsqueeze(1)
        #         .expand_as(self.cls_score2.weight.data)
        #     )
        #     self.cls_score2.weight.data = self.cls_score2.weight.data.div(
        #         temp_norm + 1e-5
        #     )
        #     cos_dist = self.cls_score2(x_normalized)
        #     scores.append(self.scale * cos_dist)
        #     proposal_deltas.append(self.bbox_pred2(x))
        
        return scores, proposal_deltas

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        losses = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            # self.box_reg_loss_type,
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


# @ROI_HEADS_OUTPUT_REGISTRY.register()
# class CosineSimOutputLayers(nn.Module):
#     """
#     Two outputs
#     (1) proposal-to-detection box regression deltas (the same as
#         the FastRCNNOutputLayers)
#     (2) classification score is based on cosine_similarity
#     """

#     def __init__(
#         self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
#     ):
#         """
#         Args:
#             cfg: config
#             input_size (int): channels, or (channels, height, width)
#             num_classes (int): number of foreground classes
#             cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
#             box_dim (int): the dimension of bounding boxes.
#                 Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
#         """
#         super(CosineSimOutputLayers, self).__init__()

#         if not isinstance(input_size, int):
#             input_size = np.prod(input_size)

#         # The prediction layer for num_classes foreground classes and one
#         # background class
#         # (hence + 1)
#         self.cls_score = nn.Linear(input_size, num_classes + 1, bias=False)
#         self.scale = cfg.MODEL.ROI_HEADS.COSINE_SCALE
#         if self.scale == -1:
#             # learnable global scaling factor
#             self.scale = nn.Parameter(torch.ones(1) * 20.0)
#         num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
#         self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

#         nn.init.normal_(self.cls_score.weight, std=0.01)
#         nn.init.normal_(self.bbox_pred.weight, std=0.001)
#         for l in [self.bbox_pred]:
#             nn.init.constant_(l.bias, 0)

#     def forward(self, x):
#         if x.dim() > 2:
#             x = torch.flatten(x, start_dim=1)

#         # normalize the input x along the `input_size` dimension
#         x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
#         x_normalized = x.div(x_norm + 1e-5)

#         # normalize weight
#         temp_norm = (
#             torch.norm(self.cls_score.weight.data, p=2, dim=1)
#             .unsqueeze(1)
#             .expand_as(self.cls_score.weight.data)
#         )
#         self.cls_score.weight.data = self.cls_score.weight.data.div(
#             temp_norm + 1e-5
#         )
#         cos_dist = self.cls_score(x_normalized)
#         scores = self.scale * cos_dist
#         proposal_deltas = self.bbox_pred(x)
#         return scores, proposal_deltas
