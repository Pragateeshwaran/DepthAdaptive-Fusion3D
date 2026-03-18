import torch

from rpn_refinement import bev_iou


CLASS_NAME_TO_ID = {
    'Pedestrian': 0,
    'Cyclist': 1,
    'Car': 2,
}


def _to_cpu_detached_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _compute_ap_11_point(recall, precision):
    ap = 0.0
    for t in torch.linspace(0.0, 1.0, steps=11):
        mask = recall >= t
        p = precision[mask].max() if mask.any() else torch.tensor(0.0)
        ap += float(p.item())
    return ap / 11.0


def _threshold_key(t):
    return str(float(t))


def _zero_result_for_threshold():
    return {'Car': 0.0, 'Pedestrian': 0.0, 'Cyclist': 0.0, 'mAP': 0.0}


def _empty_results(iou_thresholds):
    return {_threshold_key(t): _zero_result_for_threshold() for t in iou_thresholds}


def compute_map(predictions, targets, iou_thresholds=(0.5, 0.7)):
    """
    Compute per-class AP (11-point interpolation) and mean AP for one or more IoU thresholds.

    predictions: list[dict] with keys boxes (N,7), scores (N,), labels (N,)
    targets: list[dict] with keys boxes (M,7), labels (M,)
    """
    if isinstance(iou_thresholds, (float, int)):
        iou_thresholds = [float(iou_thresholds)]
    else:
        iou_thresholds = [float(t) for t in iou_thresholds]

    class_names = ['Car', 'Pedestrian', 'Cyclist']

    if len(predictions) == 0 or len(targets) == 0:
        return _empty_results(iou_thresholds)

    all_results = {}

    for iou_threshold in iou_thresholds:
        threshold_key = _threshold_key(iou_threshold)
        results = {k: 0.0 for k in class_names}
        aps = []

        for class_name in class_names:
            class_id = CLASS_NAME_TO_ID[class_name]
            total_gt = 0

            gt_by_image = {}
            for img_idx, tgt in enumerate(targets):
                gt_boxes = _to_cpu_detached_tensor(tgt.get('boxes', torch.zeros((0, 7))))
                gt_labels = _to_cpu_detached_tensor(
                    tgt.get('labels', torch.zeros((0,), dtype=torch.long))
                ).long()
                cls_mask = gt_labels == class_id
                cls_gt_boxes = gt_boxes[cls_mask] if gt_boxes.numel() > 0 else gt_boxes.new_zeros((0, 7))
                gt_by_image[img_idx] = {
                    'boxes': cls_gt_boxes,
                    'matched': torch.zeros(cls_gt_boxes.shape[0], dtype=torch.bool),
                }
                total_gt += cls_gt_boxes.shape[0]

            global_pred_records = []
            iou_rows_by_image = {}

            for img_idx, pred in enumerate(predictions):
                pred_boxes = _to_cpu_detached_tensor(pred.get('boxes', torch.zeros((0, 7))))
                pred_scores = _to_cpu_detached_tensor(pred.get('scores', torch.zeros((0,))))
                pred_labels = _to_cpu_detached_tensor(
                    pred.get('labels', torch.zeros((0,), dtype=torch.long))
                ).long()

                if pred_boxes.numel() == 0:
                    iou_rows_by_image[img_idx] = {
                        'ious': torch.zeros((0, gt_by_image[img_idx]['boxes'].shape[0]), dtype=torch.float32),
                    }
                    continue

                cls_mask = pred_labels == class_id
                cls_boxes = pred_boxes[cls_mask]
                cls_scores = pred_scores[cls_mask]

                if cls_boxes.shape[0] == 0:
                    iou_rows_by_image[img_idx] = {
                        'ious': torch.zeros((0, gt_by_image[img_idx]['boxes'].shape[0]), dtype=torch.float32),
                    }
                    continue

                sort_idx = torch.argsort(cls_scores, descending=True)
                cls_boxes = cls_boxes[sort_idx]
                cls_scores = cls_scores[sort_idx]

                gt_boxes = gt_by_image[img_idx]['boxes']
                if gt_boxes.shape[0] > 0:
                    ious = bev_iou(cls_boxes, gt_boxes).cpu()
                else:
                    ious = torch.zeros((cls_boxes.shape[0], 0), dtype=torch.float32)

                iou_rows_by_image[img_idx] = {'ious': ious}

                for row_idx in range(cls_boxes.shape[0]):
                    global_pred_records.append((cls_scores[row_idx].item(), img_idx, row_idx))

            if total_gt == 0 or len(global_pred_records) == 0:
                results[class_name] = 0.0
                aps.append(0.0)
                continue

            global_pred_records.sort(key=lambda x: x[0], reverse=True)
            tp = torch.zeros(len(global_pred_records), dtype=torch.float32)
            fp = torch.zeros(len(global_pred_records), dtype=torch.float32)

            for i, (_, img_idx, row_idx) in enumerate(global_pred_records):
                gt_info = gt_by_image[img_idx]
                gt_boxes = gt_info['boxes']

                if gt_boxes.shape[0] == 0:
                    fp[i] = 1.0
                    continue

                ious = iou_rows_by_image[img_idx]['ious'][row_idx]
                best_iou, best_gt_idx = torch.max(ious, dim=0)

                if best_iou.item() >= iou_threshold and not gt_info['matched'][best_gt_idx]:
                    tp[i] = 1.0
                    gt_info['matched'][best_gt_idx] = True
                else:
                    fp[i] = 1.0

            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)

            recall = tp_cum / max(total_gt, 1)
            precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)

            ap = _compute_ap_11_point(recall, precision)
            results[class_name] = ap
            aps.append(ap)

        results['mAP'] = float(sum(aps) / max(len(aps), 1))
        all_results[threshold_key] = results

    return all_results
