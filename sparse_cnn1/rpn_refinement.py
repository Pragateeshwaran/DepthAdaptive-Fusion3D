import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============ Utility Functions ============

def decode_boxes(anchors, deltas):
    """
    Decode box predictions using anchor boxes.
    FIXED: LiDAR coordinates (X=forward, Y=left, Z=up)
    """
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
    """
    Encode ground truth boxes relative to anchors.
    FIXED: Must match decode_boxes formula for LiDAR coordinates
    """
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


def _box_to_bev_corners(box):
    """Convert [x, y, z, h, w, l, rot] box to 4 BEV corners (4, 2)."""
    x, y, _, _, w, l, rot = box
    half_l = l * 0.5
    half_w = w * 0.5

    local = torch.stack(
        [
            torch.stack([half_l, half_w]),
            torch.stack([-half_l, half_w]),
            torch.stack([-half_l, -half_w]),
            torch.stack([half_l, -half_w]),
        ],
        dim=0,
    )

    c = torch.cos(rot)
    s = torch.sin(rot)
    rot_mat = torch.stack(
        [
            torch.stack([c, -s]),
            torch.stack([s, c]),
        ],
        dim=0,
    )

    rotated = local @ rot_mat.T
    center = torch.stack([x, y]).unsqueeze(0)
    return rotated + center


def _polygon_area(poly):
    """Shoelace area for polygon vertices (K,2)."""
    if poly.shape[0] < 3:
        return poly.new_tensor(0.0)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1) - y * torch.roll(x, shifts=-1)))


def _line_intersection(p1, p2, q1, q2):
    """Intersection of two infinite lines defined by p1->p2 and q1->q2."""
    r = p2 - p1
    s = q2 - q1
    rxs = r[0] * s[1] - r[1] * s[0]
    qp = q1 - p1
    t = (qp[0] * s[1] - qp[1] * s[0]) / (rxs + 1e-8)
    return p1 + t * r


def _inside(point, edge_start, edge_end, orientation_sign):
    """Check if point is inside half-plane for edge with polygon orientation."""
    edge = edge_end - edge_start
    rel = point - edge_start
    cross = edge[0] * rel[1] - edge[1] * rel[0]
    return (cross * orientation_sign) >= 0


def _polygon_clip(subject, clip_poly):
    """Sutherland-Hodgman polygon clipping in pure PyTorch."""
    if subject.shape[0] == 0:
        return subject

    output = subject
    clip_area = _polygon_area(clip_poly)
    orientation_sign = torch.where(clip_area > 0, clip_area.new_tensor(1.0), clip_area.new_tensor(-1.0))

    for i in range(clip_poly.shape[0]):
        cp1 = clip_poly[i]
        cp2 = clip_poly[(i + 1) % clip_poly.shape[0]]

        if output.shape[0] == 0:
            break

        input_poly = output
        new_vertices = []

        s = input_poly[-1]
        for j in range(input_poly.shape[0]):
            e = input_poly[j]
            e_inside = _inside(e, cp1, cp2, orientation_sign)
            s_inside = _inside(s, cp1, cp2, orientation_sign)

            if e_inside:
                if not s_inside:
                    inter = _line_intersection(s, e, cp1, cp2)
                    new_vertices.append(inter)
                new_vertices.append(e)
            elif s_inside:
                inter = _line_intersection(s, e, cp1, cp2)
                new_vertices.append(inter)

            s = e

        if len(new_vertices) == 0:
            output = input_poly.new_zeros((0, 2))
        else:
            output = torch.stack(new_vertices, dim=0)

    return output


def bev_iou(boxes1, boxes2):
    """
    Compute rotated Bird's Eye View IoU between boxes.

    Args:
        boxes1: (N, 7) [x, y, z, h, w, l, rot]
        boxes2: (M, 7) [x, y, z, h, w, l, rot]

    Returns:
        IoU matrix of shape (N, M)
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    if N == 0 or M == 0:
        return boxes1.new_zeros((N, M))

    iou = boxes1.new_zeros((N, M))

    for i in range(N):
        poly1 = _box_to_bev_corners(boxes1[i])
        area1 = _polygon_area(poly1)

        for j in range(M):
            poly2 = _box_to_bev_corners(boxes2[j])
            area2 = _polygon_area(poly2)

            inter_poly = _polygon_clip(poly1, poly2)
            inter_area = _polygon_area(inter_poly)

            union = area1 + area2 - inter_area
            iou[i, j] = inter_area / (union + 1e-6)

    return iou


def nms_bev(boxes, scores, iou_threshold=0.5):
    """Non-Maximum Suppression in Bird's Eye View."""
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    _, order = scores.sort(descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        i = order[0].item()
        keep.append(i)

        ious = bev_iou(boxes[i:i+1], boxes[order[1:]])
        mask = ious[0] <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


class AnchorGenerator:
    """Generate 3D anchor boxes."""
    def __init__(self,
                 anchor_sizes=[
                     (1.8, 0.6, 0.6),
                     (1.6, 0.7, 1.7),
                     (1.5, 1.6, 3.9)
                 ],
                 anchor_rotations=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                 feature_map_size=(200, 200),
                 voxel_size=(0.5, 0.5, 0.5),
                 point_cloud_range=(-50, -50, -5, 50, 50, 10)):

        self.anchor_sizes = anchor_sizes
        self.anchor_rotations = anchor_rotations
        self.feature_map_size = feature_map_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        self.num_anchors_per_location = len(anchor_sizes) * len(anchor_rotations)

    def generate_anchors(self, device='cuda', feature_map_size=None):
        """Generate anchors for entire feature map."""
        if feature_map_size is not None:
            H, W = feature_map_size
        else:
            H, W = self.feature_map_size

        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        x_step = (x_max - x_min) / float(W)
        y_step = (y_max - y_min) / float(H)
        x_range = x_min + (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * x_step
        y_range = y_min + (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * y_step

        x_grid, y_grid = torch.meshgrid(x_range, y_range, indexing='xy')
        z_centers = torch.full_like(x_grid, -1.0)

        anchors_list = []

        for size in self.anchor_sizes:
            h, w, l = size
            for rotation in self.anchor_rotations:
                anchor = torch.stack([
                    x_grid, y_grid, z_centers,
                    torch.full_like(x_grid, h),
                    torch.full_like(x_grid, w),
                    torch.full_like(x_grid, l),
                    torch.full_like(x_grid, rotation)
                ], dim=-1)
                anchors_list.append(anchor)

        anchors = torch.stack(anchors_list, dim=2)
        return anchors


class RPNHead(nn.Module):
    """Single-stage RPN head with objectness + multiclass logits + box deltas."""
    def __init__(self, in_channels=128, num_anchors_per_location=12, num_classes=3):
        super(RPNHead, self).__init__()

        self.num_anchors = num_anchors_per_location
        self.num_classes = num_classes

        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_obj = nn.Conv2d(256, num_anchors_per_location, 1)
        self.conv_cls_multiclass = nn.Conv2d(256, num_anchors_per_location * num_classes, 1)
        self.conv_reg = nn.Conv2d(256, num_anchors_per_location * 7, 1)

    def forward(self, features):
        B, _, H, W = features.shape
        x = self.conv_shared(features)

        obj_preds = self.conv_obj(x)
        obj_preds = obj_preds.permute(0, 2, 3, 1).contiguous()

        cls_multi_preds = self.conv_cls_multiclass(x)
        cls_multi_preds = cls_multi_preds.permute(0, 2, 3, 1).contiguous()

        box_preds = self.conv_reg(x)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.view(B, H, W, self.num_anchors, 7)

        return obj_preds, cls_multi_preds, box_preds


class ProposalGenerator:
    """Generate proposals from RPN outputs."""
    def __init__(self,
                 pre_nms_top_n_train=1000,
                 pre_nms_top_n_test=500,
                 post_nms_top_n_train=300,
                 post_nms_top_n_test=100,
                 nms_thresh=0.3,
                 score_thresh=0.001,
                 num_classes=3):

        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.num_classes = num_classes

    def __call__(self, anchors, obj_preds, cls_multi_preds, box_preds, training=True):
        B, H, W, num_anchors = obj_preds.shape

        anchors_flat = anchors.view(-1, 7)

        pre_nms_top_n = self.pre_nms_top_n_train if training else self.pre_nms_top_n_test
        post_nms_top_n = self.post_nms_top_n_train if training else self.post_nms_top_n_test

        proposals_list = []
        scores_list = []
        labels_list = []

        for b in range(B):
            obj_scores = torch.sigmoid(obj_preds[b]).view(-1)
            cls_scores = torch.sigmoid(cls_multi_preds[b].view(-1, self.num_classes))
            max_cls_scores, cls_labels = cls_scores.max(dim=1)
            scores = obj_scores * max_cls_scores

            box_deltas = box_preds[b].view(-1, 7)

            top_n = min(pre_nms_top_n, scores.shape[0])
            top_scores, top_indices = scores.topk(top_n)

            selected_anchors = anchors_flat[top_indices]
            selected_deltas = box_deltas[top_indices]
            proposals = decode_boxes(selected_anchors, selected_deltas)
            selected_labels = cls_labels[top_indices]

            keep = top_scores >= self.score_thresh
            proposals = proposals[keep]
            top_scores = top_scores[keep]
            selected_labels = selected_labels[keep]

            if proposals.shape[0] > 0:
                keep_indices = nms_bev(proposals, top_scores, self.nms_thresh)
                proposals = proposals[keep_indices]
                top_scores = top_scores[keep_indices]
                selected_labels = selected_labels[keep_indices]

                if proposals.shape[0] > post_nms_top_n:
                    proposals = proposals[:post_nms_top_n]
                    top_scores = top_scores[:post_nms_top_n]
                    selected_labels = selected_labels[:post_nms_top_n]

            proposals_list.append(proposals)
            scores_list.append(top_scores)
            labels_list.append(selected_labels)

        return proposals_list, scores_list, labels_list


class RPNLoss(nn.Module):
    """Single-stage loss: objectness focal + multiclass focal + box regression."""
    def __init__(
        self,
        num_classes=3,
        pos_iou_thresh=0.25,
        neg_iou_thresh=0.10,
        samples_per_image=256,
        positive_fraction=0.5,
        focal_alpha=0.25,
        focal_gamma=2.0,
        reg_beta=1.0,
    ):
        super(RPNLoss, self).__init__()
        self.num_classes = num_classes
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.samples_per_image = samples_per_image
        self.positive_fraction = positive_fraction
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.reg_beta = reg_beta

    def assign_targets(self, anchors, gt_boxes, gt_labels):
        """Assign objectness/class/regression targets."""
        num_anchors = anchors.shape[0]
        if gt_boxes.shape[0] == 0:
            labels = torch.zeros(num_anchors, device=anchors.device, dtype=torch.long)
            matched_boxes = torch.zeros_like(anchors)
            matched_classes = torch.zeros(num_anchors, device=anchors.device, dtype=torch.long)
            return labels, matched_boxes, matched_classes

        ious = bev_iou(anchors, gt_boxes)
        max_ious, max_indices = ious.max(dim=1)

        labels = torch.full((num_anchors,), -1, device=anchors.device, dtype=torch.long)
        labels[max_ious < self.neg_iou_thresh] = 0
        labels[max_ious >= self.pos_iou_thresh] = 1

        _, gt_best_anchor_idx = ious.max(dim=0)
        labels[gt_best_anchor_idx] = 1
        max_indices[gt_best_anchor_idx] = torch.arange(
            gt_boxes.shape[0], device=anchors.device, dtype=max_indices.dtype
        )

        matched_boxes = gt_boxes[max_indices]
        matched_classes = gt_labels[max_indices]
        return labels, matched_boxes, matched_classes

    def _sample_anchors(self, labels):
        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == 0)[0]

        num_pos = min(int(self.samples_per_image * self.positive_fraction), pos_idx.numel())
        num_neg = min(self.samples_per_image - num_pos, neg_idx.numel())

        if pos_idx.numel() > num_pos:
            pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=labels.device)[:num_pos]]
        if neg_idx.numel() > num_neg:
            neg_idx = neg_idx[torch.randperm(neg_idx.numel(), device=labels.device)[:num_neg]]

        return pos_idx, neg_idx

    def _sigmoid_focal_loss(self, logits, targets):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        mod = (1.0 - p_t).pow(self.focal_gamma)
        return (alpha_t * mod * ce).mean() if ce.numel() > 0 else logits.new_tensor(0.0)

    def forward(self, anchors, obj_preds, cls_multi_preds, box_preds, targets):
        B = obj_preds.shape[0]
        device = obj_preds.device

        anchors_flat = anchors.view(-1, 7)
        obj_preds_flat = obj_preds.view(B, -1)
        cls_preds_flat = cls_multi_preds.view(B, -1, self.num_classes)
        box_preds_flat = box_preds.view(B, -1, 7)

        total_obj_loss = obj_preds.new_tensor(0.0)
        total_multi_cls_loss = obj_preds.new_tensor(0.0)
        total_reg_loss = obj_preds.new_tensor(0.0)
        num_pos = 0

        for b in range(B):
            gt_boxes = targets[b]['boxes_3d']
            gt_labels = targets[b]['labels']

            labels, target_boxes, target_classes = self.assign_targets(anchors_flat, gt_boxes, gt_labels)
            pos_idx, neg_idx = self._sample_anchors(labels)
            sample_idx = torch.cat([pos_idx, neg_idx], dim=0)

            if sample_idx.numel() > 0:
                obj_logits = obj_preds_flat[b][sample_idx]
                obj_targets = labels[sample_idx].float()
                total_obj_loss += self._sigmoid_focal_loss(obj_logits, obj_targets)

            if pos_idx.numel() > 0:
                pos_anchors = anchors_flat[pos_idx]
                pos_box_preds = box_preds_flat[b][pos_idx]
                pos_targets = target_boxes[pos_idx]

                target_deltas = encode_boxes(pos_targets, pos_anchors)
                target_deltas[:, 6] = torch.atan2(
                    torch.sin(target_deltas[:, 6]), torch.cos(target_deltas[:, 6])
                )
                target_deltas = torch.clamp(target_deltas, min=-10.0, max=10.0)

                reg_loss = F.smooth_l1_loss(
                    pos_box_preds, target_deltas, beta=self.reg_beta, reduction='sum'
                ) / max(1, pos_idx.numel())
                total_reg_loss += reg_loss

                class_logits = cls_preds_flat[b][pos_idx]
                class_targets = F.one_hot(
                    target_classes[pos_idx].clamp(min=0, max=self.num_classes - 1),
                    num_classes=self.num_classes,
                ).float()
                total_multi_cls_loss += self._sigmoid_focal_loss(class_logits, class_targets)

                num_pos += pos_idx.numel()

        if B > 0:
            total_obj_loss = total_obj_loss / B
            total_multi_cls_loss = total_multi_cls_loss / B
            total_reg_loss = total_reg_loss / B

        return {
            'rpn_cls_loss': total_obj_loss,
            'rpn_multi_cls_loss': total_multi_cls_loss,
            'rpn_reg_loss': total_reg_loss,
            'num_pos_anchors': num_pos / B if B > 0 else 0,
        }


class SingleStageDetector(nn.Module):
    """Single-stage detector with objectness, multiclass scores, and box deltas."""
    def __init__(self, backbone_channels=128, num_classes=3, num_anchors_per_location=12,
                 pos_iou_thresh=0.25, neg_iou_thresh=0.10):
        super(SingleStageDetector, self).__init__()

        self.num_classes = num_classes
        self.anchor_generator = AnchorGenerator()
        self.rpn_head = RPNHead(
            backbone_channels,
            num_anchors_per_location=num_anchors_per_location,
            num_classes=num_classes,
        )
        self.proposal_generator = ProposalGenerator(num_classes=num_classes)
        self.rpn_loss_fn = RPNLoss(
            num_classes=num_classes,
            pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh,
        )

    def forward(self, features, targets=None, training=True):
        device = features.device
        _, _, H_feat, W_feat = features.shape

        anchors = self.anchor_generator.generate_anchors(device, feature_map_size=(H_feat, W_feat))

        obj_preds, cls_multi_preds, box_preds = self.rpn_head(features)

        proposals_list, scores_list, labels_list = self.proposal_generator(
            anchors, obj_preds, cls_multi_preds, box_preds, training
        )

        output = {
            'objectness_scores': obj_preds,
            'class_scores': cls_multi_preds,
            'box_deltas': box_preds,
            'proposals': proposals_list,
            'scores': scores_list,
            'labels': labels_list,
        }

        if training and targets is not None:
            output['losses'] = self.rpn_loss_fn(anchors, obj_preds, cls_multi_preds, box_preds, targets)

        return output


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
