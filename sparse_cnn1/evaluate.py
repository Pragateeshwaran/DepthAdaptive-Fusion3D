import torch

from rpn_refinement import bev_iou


CLASS_NAME_BY_LABEL = {
    0: 'Pedestrian',
    1: 'Cyclist',
    2: 'Car',
}


def _to_cpu_tensor(x, shape_hint=None, dtype=torch.float32):
    if x is None:
        if shape_hint is None:
            return torch.zeros((0,), dtype=dtype)
        return torch.zeros(shape_hint, dtype=dtype)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    t = torch.as_tensor(x)
    return t.detach().cpu()


def _compute_ap_11_point(tp, fp, num_gt):
    if num_gt == 0:
        return 0.0

    tp = tp.float()
    fp = fp.float()

    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)

    recalls = tp_cum / max(float(num_gt), 1.0)
    precisions = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-6)

    ap = 0.0
    for thr in torch.linspace(0.0, 1.0, steps=11):
        valid = recalls >= thr
        p = precisions[valid].max() if valid.any() else torch.tensor(0.0)
        ap += float(p)

    return ap / 11.0


def compute_map(predictions, targets, iou_threshold):
    """
    Compute per-class AP (11-point interpolation) and mean AP.

    Args:
        predictions: list[dict] with keys boxes (N,7), scores (N,), labels (N,)
        targets: list[dict] with keys boxes (M,7), labels (M,)
        iou_threshold: float

    Returns:
        dict with keys Car, Pedestrian, Cyclist, mAP
    """
    class_ids = [0, 1, 2]
    ap_by_name = {'Car': 0.0, 'Pedestrian': 0.0, 'Cyclist': 0.0}

    for cls_id in class_ids:
        num_gt = 0
        gt_records = []
        pred_records = []

        for sample_idx, target in enumerate(targets):
            t_boxes = _to_cpu_tensor(target.get('boxes'), shape_hint=(0, 7), dtype=torch.float32)
            t_labels = _to_cpu_tensor(target.get('labels'), shape_hint=(0,), dtype=torch.long)

            gt_mask = (t_labels == cls_id)
            gt_boxes_cls = t_boxes[gt_mask] if t_boxes.numel() > 0 else t_boxes.view(0, 7)
            num_gt += int(gt_boxes_cls.shape[0])
            gt_records.append({'boxes': gt_boxes_cls, 'matched': torch.zeros(gt_boxes_cls.shape[0], dtype=torch.bool)})

            pred = predictions[sample_idx] if sample_idx < len(predictions) else {}
            p_boxes = _to_cpu_tensor(pred.get('boxes'), shape_hint=(0, 7), dtype=torch.float32)
            p_scores = _to_cpu_tensor(pred.get('scores'), shape_hint=(0,), dtype=torch.float32)
            p_labels = _to_cpu_tensor(pred.get('labels'), shape_hint=(0,), dtype=torch.long)

            pred_mask = (p_labels == cls_id)
            p_boxes_cls = p_boxes[pred_mask] if p_boxes.numel() > 0 else p_boxes.view(0, 7)
            p_scores_cls = p_scores[pred_mask] if p_scores.numel() > 0 else p_scores.view(0)

            for box, score in zip(p_boxes_cls, p_scores_cls):
                pred_records.append({'sample_idx': sample_idx, 'box': box, 'score': float(score)})

        if len(pred_records) == 0:
            ap_by_name[CLASS_NAME_BY_LABEL[cls_id]] = 0.0
            continue

        pred_records.sort(key=lambda r: r['score'], reverse=True)

        tp = torch.zeros(len(pred_records), dtype=torch.float32)
        fp = torch.zeros(len(pred_records), dtype=torch.float32)

        for i, pred in enumerate(pred_records):
            sample_idx = pred['sample_idx']
            gt_boxes = gt_records[sample_idx]['boxes']
            gt_matched = gt_records[sample_idx]['matched']

            if gt_boxes.shape[0] == 0:
                fp[i] = 1.0
                continue

            ious = bev_iou(pred['box'].unsqueeze(0), gt_boxes).squeeze(0)
            best_iou, best_idx = ious.max(dim=0)

            if best_iou >= iou_threshold and not gt_matched[best_idx]:
                tp[i] = 1.0
                gt_matched[best_idx] = True
            else:
                fp[i] = 1.0

        ap_by_name[CLASS_NAME_BY_LABEL[cls_id]] = _compute_ap_11_point(tp, fp, num_gt)

    ap_values = [ap_by_name['Car'], ap_by_name['Pedestrian'], ap_by_name['Cyclist']]
    ap_by_name['mAP'] = float(sum(ap_values) / len(ap_values))
    return ap_by_name
