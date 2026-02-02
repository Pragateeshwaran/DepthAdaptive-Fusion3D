import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============ Utility Functions ============

def decode_boxes(anchors, deltas):
    """Decode box predictions using anchor boxes."""
    xa, ya, za, ha, wa, la, rota = anchors.unbind(dim=-1)
    dx, dy, dz, dh, dw, dl, drot = deltas.unbind(dim=-1)
    
    x = xa + dx * wa
    y = ya + dy * ha
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
    
    dx = (x - xa) / (wa + 1e-6)
    dy = (y - ya) / (ha + 1e-6)
    dz = (z - za) / (ha + 1e-6)
    
    dh = torch.log(h / (ha + 1e-6))
    dw = torch.log(w / (wa + 1e-6))
    dl = torch.log(l / (la + 1e-6))
    
    drot = rot - rota
    
    return torch.stack([dx, dy, dz, dh, dw, dl, drot], dim=-1)


def bev_iou(boxes1, boxes2):
    """
    Compute Bird's Eye View IoU between boxes.
    Simplified version - assumes axis-aligned boxes for speed.
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    
    # Extract BEV parameters (x, y, w, l)
    x1, y1 = boxes1[:, 0], boxes1[:, 1]
    w1, l1 = boxes1[:, 4], boxes1[:, 5]
    
    x2, y2 = boxes2[:, 0], boxes2[:, 1]
    w2, l2 = boxes2[:, 4], boxes2[:, 5]
    
    # Compute areas
    area1 = w1 * l1  # (N,)
    area2 = w2 * l2  # (M,)
    
    # Expand for broadcasting
    x1 = x1.unsqueeze(1)  # (N, 1)
    y1 = y1.unsqueeze(1)
    w1 = w1.unsqueeze(1)
    l1 = l1.unsqueeze(1)
    area1 = area1.unsqueeze(1)
    
    x2 = x2.unsqueeze(0)  # (1, M)
    y2 = y2.unsqueeze(0)
    w2 = w2.unsqueeze(0)
    l2 = l2.unsqueeze(0)
    area2 = area2.unsqueeze(0)
    
    # Compute intersection (axis-aligned approximation)
    x_overlap = torch.max(torch.zeros_like(x1), 
                         torch.min(x1 + w1/2, x2 + w2/2) - torch.max(x1 - w1/2, x2 - w2/2))
    y_overlap = torch.max(torch.zeros_like(y1),
                         torch.min(y1 + l1/2, y2 + l2/2) - torch.max(y1 - l1/2, y2 - l2/2))
    
    intersection = x_overlap * y_overlap
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-6)
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
                     (1.8, 0.6, 0.6),   # Pedestrian: (h, w, l)
                     (1.6, 0.7, 1.7),   # Cyclist: (h, w, l)
                     (1.5, 1.6, 3.9)    # Car: (h, w, l)
                 ],
                 anchor_rotations=[0, np.pi/2],
                 feature_map_size=(200, 200),
                 voxel_size=(0.5, 0.5, 0.5),
                 point_cloud_range=(-50, -50, -3, 50, 50, 5)):
        
        self.anchor_sizes = anchor_sizes
        self.anchor_rotations = anchor_rotations
        self.feature_map_size = feature_map_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        
        self.num_anchors_per_location = len(anchor_sizes) * len(anchor_rotations)
    
    def generate_anchors(self, device='cuda', feature_map_size=None):
        """Generate anchors for entire feature map.
        
        Args:
            device: Device to create anchors on
            feature_map_size: (H, W) tuple. If None, uses self.feature_map_size
        """
        if feature_map_size is not None:
            H, W = feature_map_size
        else:
            H, W = self.feature_map_size
        
        # Create grid centers
        x_range = torch.linspace(self.point_cloud_range[0], 
                                self.point_cloud_range[3], W, device=device)
        y_range = torch.linspace(self.point_cloud_range[1], 
                                self.point_cloud_range[4], H, device=device)
        
        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing='ij')
        z_centers = torch.full_like(x_grid, -1.0)  # Ground height
        
        anchors_list = []
        
        for size in self.anchor_sizes:
            h, w, l = size
            for rotation in self.anchor_rotations:
                # Create anchor: [x, y, z, h, w, l, rot]
                anchor = torch.stack([
                    x_grid, y_grid, z_centers,
                    torch.full_like(x_grid, h),
                    torch.full_like(x_grid, w),
                    torch.full_like(x_grid, l),
                    torch.full_like(x_grid, rotation)
                ], dim=-1)  # (H, W, 7)
                
                anchors_list.append(anchor)
        
        # Stack: (H, W, num_anchors, 7)
        anchors = torch.stack(anchors_list, dim=2)
        return anchors


class RPNHead(nn.Module):
    """Region Proposal Network Head."""
    def __init__(self, in_channels=128, num_anchors_per_location=6):  # 3 sizes × 2 rotations = 6
        super(RPNHead, self).__init__()
        
        self.num_anchors = num_anchors_per_location
        
        # Shared conv
        self.conv_shared = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Objectness classification
        self.conv_cls = nn.Conv2d(256, num_anchors_per_location, 1)
        
        # Box regression (7 params)
        self.conv_reg = nn.Conv2d(256, num_anchors_per_location * 7, 1)
        
        # Direction classification (2 bins)
        self.conv_dir = nn.Conv2d(256, num_anchors_per_location * 2, 1)
    
    def forward(self, features):
        """Forward pass."""
        B, _, H, W = features.shape
        
        x = self.conv_shared(features)
        
        # Classification
        cls_preds = self.conv_cls(x)  # (B, num_anchors, H, W)
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # (B, H, W, num_anchors)
        
        # Regression
        box_preds = self.conv_reg(x)  # (B, num_anchors*7, H, W)
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # (B, H, W, num_anchors*7)
        box_preds = box_preds.view(B, H, W, self.num_anchors, 7)
        
        # Direction
        dir_preds = self.conv_dir(x)  # (B, num_anchors*2, H, W)
        dir_preds = dir_preds.permute(0, 2, 3, 1).contiguous()
        dir_preds = dir_preds.view(B, H, W, self.num_anchors, 2)
        
        return cls_preds, box_preds, dir_preds


class ProposalGenerator:
    """Generate proposals from RPN outputs."""
    def __init__(self, 
                 pre_nms_top_n_train=1000,        # IMPROVED: Reduced from 2000
                 pre_nms_top_n_test=500,          # IMPROVED: Reduced from 1000
                 post_nms_top_n_train=300,        # IMPROVED: Increased from 256
                 post_nms_top_n_test=100,         # IMPROVED: Balanced (for undertrained model)
                 nms_thresh=0.3,                  # IMPROVED: Stricter from 0.7 (fixes overlapping boxes!)
                 score_thresh=0.05):              # IMPROVED: Low for epoch 4, increase to 0.2 after epoch 30+
        
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_train = post_nms_top_n_train
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
    
    def __call__(self, anchors, cls_preds, box_preds, training=True):
        """Generate proposals."""
        B, H, W, num_anchors = cls_preds.shape
        device = cls_preds.device
        
        # Flatten anchors
        anchors_flat = anchors.view(-1, 7)  # (H*W*num_anchors, 7)
        
        # Get thresholds
        pre_nms_top_n = self.pre_nms_top_n_train if training else self.pre_nms_top_n_test
        post_nms_top_n = self.post_nms_top_n_train if training else self.post_nms_top_n_test
        
        proposals_list = []
        scores_list = []
        
        for b in range(B):
            # Get scores
            scores = torch.sigmoid(cls_preds[b]).view(-1)  # (H*W*num_anchors,)
            box_deltas = box_preds[b].view(-1, 7)  # (H*W*num_anchors, 7)
            
            # Pre-NMS filtering
            top_n = min(pre_nms_top_n, scores.shape[0])
            top_scores, top_indices = scores.topk(top_n)
            
            # Decode boxes
            selected_anchors = anchors_flat[top_indices]
            selected_deltas = box_deltas[top_indices]
            proposals = decode_boxes(selected_anchors, selected_deltas)
            
            # Score threshold
            keep = top_scores >= self.score_thresh
            proposals = proposals[keep]
            top_scores = top_scores[keep]
            
            # NMS
            if proposals.shape[0] > 0:
                keep_indices = nms_bev(proposals, top_scores, self.nms_thresh)
                proposals = proposals[keep_indices]
                top_scores = top_scores[keep_indices]
                
                # Post-NMS top-k
                if proposals.shape[0] > post_nms_top_n:
                    proposals = proposals[:post_nms_top_n]
                    top_scores = top_scores[:post_nms_top_n]
            
            proposals_list.append(proposals)
            scores_list.append(top_scores)
        
        return proposals_list, scores_list


class RPNLoss(nn.Module):
    """RPN Loss computation with IMPROVED IoU thresholds."""
    def __init__(self, pos_iou_thresh=0.5, neg_iou_thresh=0.3):  # IMPROVED: Higher from 0.3/0.15 for better quality
        super(RPNLoss, self).__init__()
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        
        print(f"⚙️  RPN Loss initialized with IMPROVED thresholds:")
        print(f"   Positive IoU threshold: {pos_iou_thresh} (higher quality matches)")
        print(f"   Negative IoU threshold: {neg_iou_thresh} (better discrimination)")
    
    def assign_targets(self, anchors, gt_boxes):
        """Assign ground truth to anchors based on IoU."""
        if len(gt_boxes) == 0:
            labels = torch.zeros(anchors.shape[0], device=anchors.device, dtype=torch.long)
            return labels, torch.zeros_like(anchors)
        
        # Compute IoU
        ious = bev_iou(anchors, gt_boxes)  # (N, M)
        
        # Get best IoU for each anchor
        max_ious, max_indices = ious.max(dim=1)
        
        # Assign labels
        labels = torch.full((anchors.shape[0],), -1, device=anchors.device, dtype=torch.long)
        labels[max_ious < self.neg_iou_thresh] = 0  # Negative
        labels[max_ious >= self.pos_iou_thresh] = 1  # Positive
        
        # Assign target boxes
        target_boxes = gt_boxes[max_indices]
        
        return labels, target_boxes
    
    def forward(self, anchors, cls_preds, box_preds, targets):
        """Compute RPN loss."""
        B, H, W, num_anchors = cls_preds.shape
        device = cls_preds.device
        
        # Flatten
        anchors_flat = anchors.view(-1, 7)  # (H*W*num_anchors, 7)
        cls_preds_flat = cls_preds.view(B, -1)  # (B, H*W*num_anchors)
        box_preds_flat = box_preds.view(B, -1, 7)  # (B, H*W*num_anchors, 7)
        
        total_cls_loss = 0
        total_reg_loss = 0
        num_pos = 0
        
        for b in range(B):
            gt_boxes = targets[b]['boxes_3d']
            
            if len(gt_boxes) == 0:
                continue
            
            # Assign targets
            labels, target_boxes = self.assign_targets(anchors_flat, gt_boxes)
            
            # Classification loss
            pos_mask = labels == 1
            neg_mask = labels == 0
            valid_mask = pos_mask | neg_mask
            
            if valid_mask.sum() > 0:
                cls_targets = labels[valid_mask].float()
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_preds_flat[b][valid_mask],
                    cls_targets,
                    reduction='mean'
                )
                total_cls_loss += cls_loss
            
            # Regression loss - only for positive anchors
            if pos_mask.sum() > 0:
                pos_anchors = anchors_flat[pos_mask]
                pos_box_preds = box_preds_flat[b][pos_mask]
                pos_targets = target_boxes[pos_mask]
                
                # Encode targets
                target_deltas = encode_boxes(pos_targets, pos_anchors)
                
                # Smooth L1 loss
                reg_loss = F.smooth_l1_loss(pos_box_preds, target_deltas, reduction='mean')
                total_reg_loss += reg_loss
                num_pos += pos_mask.sum().item()
        
        # Average over batch
        total_cls_loss = total_cls_loss / B if B > 0 else torch.tensor(0.0, device=device)
        total_reg_loss = total_reg_loss / B if B > 0 else torch.tensor(0.0, device=device)
        
        return {
            'rpn_cls_loss': total_cls_loss,
            'rpn_reg_loss': total_reg_loss,
            'num_pos_anchors': num_pos / B if B > 0 else 0
        }


class TwoStageDetector(nn.Module):
    """Complete two-stage detector with proper RPN."""
    def __init__(self, backbone_channels=128, num_classes=3, num_anchors_per_location=6,
                 pos_iou_thresh=0.3, neg_iou_thresh=0.15):
        super(TwoStageDetector, self).__init__()
        
        self.anchor_generator = AnchorGenerator()
        self.rpn_head = RPNHead(backbone_channels, num_anchors_per_location=num_anchors_per_location)
        self.proposal_generator = ProposalGenerator()
        self.rpn_loss_fn = RPNLoss(pos_iou_thresh=pos_iou_thresh, neg_iou_thresh=neg_iou_thresh)
    
    def forward(self, features, targets=None, training=True):
        """Forward pass."""
        device = features.device
        B, C, H_feat, W_feat = features.shape
        
        # Generate anchors matching the feature map size
        anchors = self.anchor_generator.generate_anchors(device, feature_map_size=(H_feat, W_feat))
        
        # RPN forward
        cls_preds, box_preds, dir_preds = self.rpn_head(features)
        
        # Generate proposals
        proposals_list, scores_list = self.proposal_generator(
            anchors, cls_preds, box_preds, training
        )
        
        if training and targets is not None:
            # Compute RPN loss
            loss_dict = self.rpn_loss_fn(anchors, cls_preds, box_preds, targets)
            
            return {
                'proposals': proposals_list,
                'scores': scores_list,
                'losses': loss_dict
            }
        else:
            return {
                'proposals': proposals_list,
                'scores': scores_list
            }


def parse_label_line(line):
    """Parse a single label line."""
    parts = line.strip().split()
    
    return {
        'type': parts[0],
        'truncated': float(parts[1]),
        'occluded': int(parts[2]),
        'alpha': float(parts[3]),
        'bbox_2d': [float(x) for x in parts[4:8]],
        'dimensions': [float(x) for x in parts[8:11]],  # [h, w, l]
        'location': [float(x) for x in parts[11:14]],   # [x, y, z]
        'rotation_y': float(parts[14]),
        'score': float(parts[15]) if len(parts) > 15 else 1.0
    }