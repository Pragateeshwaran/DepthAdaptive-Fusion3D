import torch
import torch.nn as nn
import torch.nn.functional as F


# ============ Utility Functions ============

def decode_boxes(anchors, deltas):
    """Decode box predictions using anchor boxes."""
    xa, ya, za, ha, wa, la, rota = anchors.unbind(dim=-1)
    dx, dy, dz, dh, dw, dl, drot = deltas.unbind(dim=-1)

    x = xa + dx * la
    y = ya + dy * wa
    z = za + dz * ha

    h = ha * torch.exp(dh)
    w = wa * torch.exp(dw)
    l = la * torch.exp(dl)
    rot = rota + drot

    return torch.stack([x, y, z, h, w, l, rot], dim=-1)


def encode_boxes(boxes, anchors):
    """Encode ground truth boxes relative to anchors."""
    xa, ya, za, ha, wa, la, rota = anchors.unbind(dim=-1)
    x, y, z, h, w, l, rot = boxes.unbind(dim=-1)

    dx = (x - xa) / (la + 1e-6)
    dy = (y - ya) / (wa + 1e-6)
    dz = (z - za) / (ha + 1e-6)

    dh = torch.log(h / (ha + 1e-6))
    dw = torch.log(w / (wa + 1e-6))
    dl = torch.log(l / (la + 1e-6))
    drot = rot - rota

    return torch.stack([dx, dy, dz, dh, dw, dl, drot], dim=-1)


def _boxes_to_bev_corners(boxes):
    """Convert [x, y, z, h, w, l, rot] boxes to BEV corners (N, 4, 2)."""
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4, 2))

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 4]
    l = boxes[:, 5]
    rot = boxes[:, 6]

    template = boxes.new_tensor([
        [0.5, 0.5],
        [0.5, -0.5],
        [-0.5, -0.5],
        [-0.5, 0.5],
    ])

    corners_local = template.unsqueeze(0).repeat(boxes.shape[0], 1, 1)
    corners_local[:, :, 0] = corners_local[:, :, 0] * l.unsqueeze(1)
    corners_local[:, :, 1] = corners_local[:, :, 1] * w.unsqueeze(1)

    cos_r = torch.cos(rot)
    sin_r = torch.sin(rot)

    x_local = corners_local[:, :, 0]
    y_local = corners_local[:, :, 1]

    x_rot = x_local * cos_r.unsqueeze(1) - y_local * sin_r.unsqueeze(1)
    y_rot = x_local * sin_r.unsqueeze(1) + y_local * cos_r.unsqueeze(1)

    corners = torch.stack([x_rot + x.unsqueeze(1), y_rot + y.unsqueeze(1)], dim=-1)
    return corners


def _cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _is_inside(point, edge_start, edge_end):
    return _cross_2d(edge_end - edge_start, point - edge_start) >= 0.0


def _line_intersection(p1, p2, a, b):
    r = p2 - p1
    s = b - a
    denom = _cross_2d(r, s)
    eps = p1.new_tensor(1e-8)
    t = _cross_2d(a - p1, s) / (denom + torch.sign(denom) * eps + (denom == 0).float() * eps)
    return p1 + t * r


def _polygon_clip(subject_polygon, clip_polygon):
    """Sutherland-Hodgman polygon clipping for convex polygons in pure PyTorch."""
    if subject_polygon.shape[0] == 0:
        return subject_polygon

    output = subject_polygon
    for i in range(clip_polygon.shape[0]):
        input_list = output
        if input_list.shape[0] == 0:
            break

        edge_start = clip_polygon[i]
        edge_end = clip_polygon[(i + 1) % clip_polygon.shape[0]]
        new_points = []

        prev_point = input_list[-1]
        prev_inside = _is_inside(prev_point, edge_start, edge_end)

        for curr_point in input_list:
            curr_inside = _is_inside(curr_point, edge_start, edge_end)

            if curr_inside:
                if not bool(prev_inside):
                    new_points.append(_line_intersection(prev_point, curr_point, edge_start, edge_end))
                new_points.append(curr_point)
            elif bool(prev_inside):
                new_points.append(_line_intersection(prev_point, curr_point, edge_start, edge_end))

            prev_point = curr_point
            prev_inside = curr_inside

        if len(new_points) == 0:
            output = subject_polygon.new_zeros((0, 2))
        else:
            output = torch.stack(new_points, dim=0)

    return output


def _polygon_area(poly):
    if poly.shape[0] < 3:
        return poly.new_tensor(0.0)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1) - y * torch.roll(x, shifts=-1)))


def bev_iou(boxes1, boxes2):
    """
    Rotation-aware BEV IoU using polygon clipping in pure PyTorch.
    Returns IoU matrix of shape (N, M).
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    if N == 0 or M == 0:
        return boxes1.new_zeros((N, M))

    corners1 = _boxes_to_bev_corners(boxes1)
    corners2 = _boxes_to_bev_corners(boxes2)

    area1 = boxes1[:, 4] * boxes1[:, 5]
    area2 = boxes2[:, 4] * boxes2[:, 5]

    ious = boxes1.new_zeros((N, M))
    eps = boxes1.new_tensor(1e-6)

    for i in range(N):
        poly1 = corners1[i]
        a1 = area1[i]
        for j in range(M):
            poly2 = corners2[j]
            inter_poly = _polygon_clip(poly1, poly2)
            inter_area = _polygon_area(inter_poly)
            union = a1 + area2[j] - inter_area
            ious[i, j] = inter_area / (union + eps)

    return ious


def nms_bev(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression in Bird's Eye View."""
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        ious = bev_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class AnchorGenerator:
    """Generate 3D anchor boxes."""

    def __init__(
        self,
        anchor_sizes=((1.8, 0.6, 0.6), (1.6, 0.7, 1.7), (1.5, 1.6, 3.9)),
        anchor_rotations=(0.0, 0.78539816, 1.57079633, 2.35619449),
        feature_map_size=(200, 200),
        point_cloud_range=(-50, -50, -5, 50, 50, 10),
    ):
        self.anchor_sizes = anchor_sizes
        self.anchor_rotations = anchor_rotations
        self.feature_map_size = feature_map_size
        self.point_cloud_range = point_cloud_range
        self.num_anchors_per_location = len(anchor_sizes) * len(anchor_rotations)

    def generate_anchors(self, device='cuda', feature_map_size=None):
        if feature_map_size is None:
            H, W = self.feature_map_size
        else:
            H, W = feature_map_size

        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        x_step = (x_max - x_min) / float(W)
        y_step = (y_max - y_min) / float(H)

        x_range = x_min + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * x_step
        y_range = y_min + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * y_step

        x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='xy')
        z_centers = torch.full_like(x_grid, -1.0)

        anchors_list = []
        for h, w, l in self.anchor_sizes:
            for rot in self.anchor_rotations:
                anchors_list.append(
                    torch.stack(
                        [
                            x_grid,
                            y_grid,
                            z_centers,
                            torch.full_like(x_grid, h),
                            torch.full_like(x_grid, w),
                            torch.full_like(x_grid, l),
                            torch.full_like(x_grid, rot),
                        ],
                        dim=-1,
                    )
                )

        return torch.stack(anchors_list, dim=2)


class RPNHead(nn.Module):
    """Single-stage RPN head with objectness, multiclass, and box regression."""

    def __init__(self, in_channels=128, num_anchors_per_location=12, num_classes=3):
        super().__init__()
        self.num_anchors = num_anchors_per_location
        self.num_classes = num_classes

        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.conv_obj = nn.Conv2d(256, self.num_anchors, 1)
        self.conv_cls_multiclass = nn.Conv2d(256, self.num_anchors * self.num_classes, 1)
        self.conv_reg = nn.Conv2d(256, self.num_anchors * 7, 1)

    def forward(self, features):
        B, _, H, W = features.shape
        x = self.conv_shared(features)

        obj_logits = self.conv_obj(x).permute(0, 2, 3, 1).contiguous()

        cls_logits = self.conv_cls_multiclass(x).permute(0, 2, 3, 1).contiguous()
        cls_logits = cls_logits.view(B, H, W, self.num_anchors, self.num_classes)

        box_deltas = self.conv_reg(x).permute(0, 2, 3, 1).contiguous()
        box_deltas = box_deltas.view(B, H, W, self.num_anchors, 7)

        return obj_logits, cls_logits, box_deltas


class ProposalGenerator:
    """Generate proposals from RPN outputs."""

    def __init__(
        self,
        pre_nms_top_n_train=1000,
        pre_nms_top_n_test=500,
        post_nms_top_n_train=300,
        post_nms_top_n_test=100,
        nms_thresh=0.3,
        score_thresh=1e-3,
    ):
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh

    def __call__(self, anchors, obj_logits, cls_logits, box_deltas, training=True):
        B, H, W, num_anchors = obj_logits.shape
        device = obj_logits.device

        anchors_flat = anchors.view(-1, 7)
        pre_nms_top_n = self.pre_nms_top_n_train if training else self.pre_nms_top_n_test
        post_nms_top_n = self.post_nms_top_n_train if training else self.post_nms_top_n_test

        proposals_list = []
        scores_list = []
        class_scores_list = []
        labels_list = []

        for b in range(B):
            obj_scores = torch.sigmoid(obj_logits[b]).view(-1)
            cls_probs = F.softmax(cls_logits[b].view(-1, cls_logits.shape[-1]), dim=-1)
            max_cls_scores, labels = cls_probs.max(dim=-1)
            scores = obj_scores * max_cls_scores

            deltas = box_deltas[b].view(-1, 7)
            top_n = min(pre_nms_top_n, scores.shape[0])
            top_scores, top_indices = scores.topk(top_n)

            selected_anchors = anchors_flat[top_indices]
            selected_deltas = deltas[top_indices]
            proposals = decode_boxes(selected_anchors, selected_deltas)
            selected_cls_scores = cls_probs[top_indices]
            selected_labels = labels[top_indices]

            keep = top_scores >= self.score_thresh
            proposals = proposals[keep]
            top_scores = top_scores[keep]
            selected_cls_scores = selected_cls_scores[keep]
            selected_labels = selected_labels[keep]

            if proposals.shape[0] > 0:
                keep_idx = nms_bev(proposals, top_scores, self.nms_thresh)
                proposals = proposals[keep_idx]
                top_scores = top_scores[keep_idx]
                selected_cls_scores = selected_cls_scores[keep_idx]
                selected_labels = selected_labels[keep_idx]

                if proposals.shape[0] > post_nms_top_n:
                    proposals = proposals[:post_nms_top_n]
                    top_scores = top_scores[:post_nms_top_n]
                    selected_cls_scores = selected_cls_scores[:post_nms_top_n]
                    selected_labels = selected_labels[:post_nms_top_n]

            proposals_list.append(proposals)
            scores_list.append(top_scores)
            class_scores_list.append(selected_cls_scores)
            labels_list.append(selected_labels)

        return proposals_list, scores_list, class_scores_list, labels_list


class RPNLoss(nn.Module):
    """Objectness BCE + box regression + per-class focal loss."""

    def __init__(self, num_classes=3, pos_iou_thresh=0.05, neg_iou_thresh=0.01, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.alpha = alpha
        self.gamma = gamma

    def assign_targets(self, anchors, gt_boxes, gt_labels):
        if len(gt_boxes) == 0:
            obj_labels = torch.zeros(anchors.shape[0], device=anchors.device, dtype=torch.long)
            cls_labels = torch.full((anchors.shape[0],), -1, device=anchors.device, dtype=torch.long)
            return obj_labels, cls_labels, torch.zeros_like(anchors)

        ious = bev_iou(anchors, gt_boxes)
        max_ious, max_indices = ious.max(dim=1)

        obj_labels = torch.full((anchors.shape[0],), -1, device=anchors.device, dtype=torch.long)
        obj_labels[max_ious < self.neg_iou_thresh] = 0
        obj_labels[max_ious >= self.pos_iou_thresh] = 1

        gt_best_ious, gt_best_anchor_idx = ious.max(dim=0)
        obj_labels[gt_best_anchor_idx] = 1
        max_indices[gt_best_anchor_idx] = torch.arange(gt_boxes.shape[0], device=anchors.device)

        target_boxes = gt_boxes[max_indices]
        cls_labels = torch.full((anchors.shape[0],), -1, device=anchors.device, dtype=torch.long)
        pos_mask = obj_labels == 1
        if pos_mask.any():
            cls_labels[pos_mask] = gt_labels[max_indices[pos_mask]]

        return obj_labels, cls_labels, target_boxes

    def forward(self, anchors, obj_logits, cls_logits, box_deltas, targets):
        B, H, W, num_anchors = obj_logits.shape
        device = obj_logits.device

        anchors_flat = anchors.view(-1, 7)
        obj_logits_flat = obj_logits.view(B, -1)
        cls_logits_flat = cls_logits.view(B, -1, self.num_classes)
        box_deltas_flat = box_deltas.view(B, -1, 7)

        total_obj_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        total_multi_cls_loss = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(B):
            gt_boxes = targets[b]['boxes_3d']
            gt_labels = targets[b]['labels']

            obj_labels, cls_labels, target_boxes = self.assign_targets(anchors_flat, gt_boxes, gt_labels)

            valid_mask = obj_labels >= 0
            if valid_mask.any():
                obj_targets = obj_labels[valid_mask].float()
                total_obj_loss = total_obj_loss + F.binary_cross_entropy_with_logits(
                    obj_logits_flat[b][valid_mask], obj_targets, reduction='mean'
                )

            pos_mask = obj_labels == 1
            if pos_mask.any():
                pos_anchors = anchors_flat[pos_mask]
                pos_box_preds = box_deltas_flat[b][pos_mask]
                pos_targets = target_boxes[pos_mask]
                target_deltas = encode_boxes(pos_targets, pos_anchors)
                total_reg_loss = total_reg_loss + F.smooth_l1_loss(pos_box_preds, target_deltas, reduction='mean')

                pos_cls_logits = cls_logits_flat[b][pos_mask]
                pos_cls_targets = cls_labels[pos_mask]
                ce = F.cross_entropy(pos_cls_logits, pos_cls_targets, reduction='none')
                pt = torch.exp(-ce)
                focal = self.alpha * ((1.0 - pt) ** self.gamma) * ce
                total_multi_cls_loss = total_multi_cls_loss + focal.mean()

                num_pos += pos_mask.sum().item()

        denom = float(max(B, 1))
        return {
            'rpn_cls_loss': total_obj_loss / denom,
            'rpn_reg_loss': total_reg_loss / denom,
            'rpn_multi_cls_loss': total_multi_cls_loss / denom,
            'num_pos_anchors': num_pos / denom,
        }


class SingleStageDetector(nn.Module):
    """Single-stage detector with RPN-like heads."""

    def __init__(
        self,
        backbone_channels=128,
        num_classes=3,
        num_anchors_per_location=12,
        pos_iou_thresh=0.05,
        neg_iou_thresh=0.01,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.anchor_generator = AnchorGenerator()
        self.rpn_head = RPNHead(
            in_channels=backbone_channels,
            num_anchors_per_location=num_anchors_per_location,
            num_classes=num_classes,
        )
        self.proposal_generator = ProposalGenerator()
        self.rpn_loss_fn = RPNLoss(
            num_classes=num_classes,
            pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh,
        )

    def forward(self, features, targets=None, training=True):
        device = features.device
        _, _, H_feat, W_feat = features.shape

        anchors = self.anchor_generator.generate_anchors(device, feature_map_size=(H_feat, W_feat))
        obj_logits, cls_logits, box_deltas = self.rpn_head(features)

        proposals, proposal_scores, proposal_class_scores, proposal_labels = self.proposal_generator(
            anchors, obj_logits, cls_logits, box_deltas, training=training
        )

        output = {
            'objectness_scores': torch.sigmoid(obj_logits),
            'class_scores': F.softmax(cls_logits, dim=-1),
            'box_deltas': box_deltas,
            'proposals': proposals,
            'scores': proposal_scores,
            'proposal_class_scores': proposal_class_scores,
            'proposal_labels': proposal_labels,
        }

        if training and targets is not None:
            output['losses'] = self.rpn_loss_fn(anchors, obj_logits, cls_logits, box_deltas, targets)

        return output


# Backward compatibility alias
TwoStageDetector = SingleStageDetector


def parse_label_line(line):
    """Parse a single label line."""
    parts = line.strip().split()
    return {
        'type': parts[0],
        'truncated': float(parts[1]),
        'occluded': int(parts[2]),
        'alpha': float(parts[3]),
        'bbox_2d': [float(x) for x in parts[4:8]],
        'dimensions': [float(x) for x in parts[8:11]],
        'location': [float(x) for x in parts[11:14]],
        'rotation_y': float(parts[14]),
        'score': float(parts[15]) if len(parts) > 15 else 1.0,
    }
