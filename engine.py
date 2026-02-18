# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

# ====== Visualization imports (added) ======
from PIL import Image, ImageDraw

# DETR normalize constants (must match datasets/coco.py)
_DETR_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DETR_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _unnormalize(img_tensor_chw: torch.Tensor) -> torch.Tensor:
    """
    img_tensor_chw: (3,H,W) normalized float tensor
    returns: (3,H,W) in [0,1]
    """
    mean = _DETR_MEAN.to(img_tensor_chw.device)
    std = _DETR_STD.to(img_tensor_chw.device)
    return (img_tensor_chw * std + mean).clamp(0, 1)


def _draw_boxes_pil(
    img_chw_01: torch.Tensor,
    gt_xyxy: torch.Tensor,
    pred_xyxy: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    score_thr: float,
) -> Image.Image:
    """
    Draw GT boxes (green) and prediction boxes (red + label:score) on a PIL image.
    Ensures x2>=x1 and y2>=y1; skips invalid/degenerate boxes.
    img_chw_01: (3,H,W) float in [0,1]
    gt_xyxy/pred_xyxy: (N,4) xyxy in pixels
    """
    img_uint8 = (img_chw_01 * 255).to(torch.uint8).cpu()
    pil = Image.fromarray(img_uint8.permute(1, 2, 0).numpy())
    draw = ImageDraw.Draw(pil)

    def _safe_rect(b):
        x1, y1, x2, y2 = b
        # sort coords to satisfy PIL requirements
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        # skip degenerate boxes
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            return None
        return [x1, y1, x2, y2]

    # GT in green
    if gt_xyxy is not None and gt_xyxy.numel() > 0:
        for b in gt_xyxy.detach().cpu().tolist():
            rect = _safe_rect(b)
            if rect is None:
                continue
            draw.rectangle(rect, outline=(0, 255, 0), width=3)

    # Pred in red
    if pred_xyxy is not None and pred_xyxy.numel() > 0:
        for b, s, l in zip(
            pred_xyxy.detach().cpu().tolist(),
            pred_scores.detach().cpu().tolist(),
            pred_labels.detach().cpu().tolist(),
        ):
            if s < score_thr:
                continue
            rect = _safe_rect(b)
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            draw.rectangle(rect, outline=(255, 0, 0), width=3)
            draw.text((x1, y1), f"{int(l)}:{float(s):.2f}", fill=(255, 0, 0))

    return pil


def save_vis_batch(
        samples,
        targets,
        results,
        save_dir: str,
        max_images: int = 8,
        score_thr: float = 0.3,
    ) -> None:
        """
        Save visualization for a batch.

        IMPORTANT:
        - In DETR repo, targets[i]["boxes"] is typically normalized cxcywh (0~1).
        - results[i]["boxes"] from postprocessor is xyxy in pixels.
        """
        os.makedirs(save_dir, exist_ok=True)

        imgs = samples.tensors  # B,3,H,W normalized
        B = imgs.shape[0]
        n = min(B, max_images)

        for i in range(n):
            img = _unnormalize(imgs[i])

            # ---- GT: normalized cxcywh -> pixel xyxy ----
            gt = targets[i].get("boxes", None)
            if gt is None or gt.numel() == 0:
                gt_xyxy_px = torch.zeros((0, 4), device=imgs.device)
            else:
                # orig_size is (h, w)
                h, w = targets[i]["orig_size"].tolist()
                gt = gt.detach()

                # gt is cxcywh normalized (0~1)
                cx = gt[:, 0] * w
                cy = gt[:, 1] * h
                bw = gt[:, 2] * w
                bh = gt[:, 3] * h

                x1 = cx - 0.5 * bw
                y1 = cy - 0.5 * bh
                x2 = cx + 0.5 * bw
                y2 = cy + 0.5 * bh

                gt_xyxy_px = torch.stack([x1, y1, x2, y2], dim=1)
                # clamp
                gt_xyxy_px[:, 0::2].clamp_(0, w)
                gt_xyxy_px[:, 1::2].clamp_(0, h)

            # ---- Pred: already pixel xyxy ----
            pred = results[i].get("boxes", None)
            scores = results[i].get("scores", None)
            labels = results[i].get("labels", None)

            if pred is None or pred.numel() == 0:
                pred = torch.zeros((0, 4), device=imgs.device)
                scores = torch.zeros((0,), device=imgs.device)
                labels = torch.zeros((0,), dtype=torch.long, device=imgs.device)

            pil = _draw_boxes_pil(img, gt_xyxy_px, pred, scores, labels, score_thr=score_thr)

            image_id = targets[i]["image_id"]
            if isinstance(image_id, torch.Tensor):
                image_id = int(image_id.item())

            pil.save(os.path.join(save_dir, f"img_{image_id:06d}_thr{score_thr}.png"))


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # Only rank0 should write files in DDP
    is_main = utils.is_main_process()
    if is_main:
        vis_dir = os.path.join(output_dir if output_dir is not None else ".", "vis")

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        # Save GT (green) and Pred (red) visualizations for all val images
        if is_main:
            save_vis_batch(
                samples, targets, results,
                save_dir=vis_dir, max_images=len(targets), score_thr=0.05
            )

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator