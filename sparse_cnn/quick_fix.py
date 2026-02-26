"""
DIAGNOSTIC SCRIPT: Verify IoU Calculation Fix

This script compares the OLD (broken) vs NEW (fixed) IoU calculation
to demonstrate the bug and its fix.
"""

import torch
import numpy as np

print("="*80)
print("DIAGNOSTIC: IoU Calculation Bug Analysis")
print("="*80)

# ============ OLD (BROKEN) IoU Function ============
def bev_iou_OLD_BROKEN(boxes1, boxes2):
    """
    OLD BROKEN VERSION - Uses width for X-axis overlap (WRONG!)
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
    
    # ❌ BUG: Using WIDTH for X-axis overlap (should be LENGTH!)
    x_overlap = torch.max(torch.zeros_like(x1), 
                         torch.min(x1 + w1/2, x2 + w2/2) - torch.max(x1 - w1/2, x2 - w2/2))
    y_overlap = torch.max(torch.zeros_like(y1),
                         torch.min(y1 + l1/2, y2 + l2/2) - torch.max(y1 - l1/2, y2 - l2/2))
    
    intersection = x_overlap * y_overlap
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


# ============ NEW (FIXED) IoU Function ============
def bev_iou_NEW_FIXED(boxes1, boxes2):
    """
    NEW FIXED VERSION - Uses length for X-axis overlap (CORRECT!)
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
    
    # ✅ FIXED: Using LENGTH for X-axis overlap, WIDTH for Y-axis
    x_overlap = torch.max(torch.zeros_like(x1), 
                         torch.min(x1 + l1/2, x2 + l2/2) - torch.max(x1 - l1/2, x2 - l2/2))
    y_overlap = torch.max(torch.zeros_like(y1),
                         torch.min(y1 + w1/2, y2 + w2/2) - torch.max(y1 - w1/2, y2 - w2/2))
    
    intersection = x_overlap * y_overlap
    union = area1 + area2 - intersection
    
    iou = intersection / (union + 1e-6)
    return iou


# ============ Test Cases ============

print("\n" + "="*80)
print("TEST CASE 1: Perfect Match (Same Box)")
print("="*80)

# Box format: [x, y, z, h, w, l, rot]
# Car: h=1.5, w=1.6, l=3.9
box1 = torch.tensor([[0.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0]])
box2 = torch.tensor([[0.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0]])

iou_old = bev_iou_OLD_BROKEN(box1, box2)
iou_new = bev_iou_NEW_FIXED(box1, box2)

print(f"Box 1: center=(0.0, 0.0), size=(h=1.5, w=1.6, l=3.9)")
print(f"Box 2: center=(0.0, 0.0), size=(h=1.5, w=1.6, l=3.9)")
print(f"\nOLD (broken) IoU: {iou_old[0, 0]:.4f}")
print(f"NEW (fixed) IoU:  {iou_new[0, 0]:.4f}")
print(f"Expected: 1.0000 (perfect match)")

if abs(iou_new[0, 0] - 1.0) < 0.001:
    print("✅ PASS: Fixed version gives perfect IoU")
else:
    print("❌ FAIL: Fixed version still incorrect")

print("\n" + "="*80)
print("TEST CASE 2: Partial Overlap in X-direction")
print("="*80)

# Two cars with partial X overlap
# Car 1 at x=0, Car 2 at x=2 (2m forward)
# Length = 3.9m, so overlap should exist
box1 = torch.tensor([[0.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0]])
box2 = torch.tensor([[2.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0]])

iou_old = bev_iou_OLD_BROKEN(box1, box2)
iou_new = bev_iou_NEW_FIXED(box1, box2)

print(f"Box 1: center=(0.0, 0.0), size=(h=1.5, w=1.6, l=3.9)")
print(f"Box 2: center=(2.0, 0.0), size=(h=1.5, w=1.6, l=3.9)")
print(f"\nX-extent Box 1: [-1.95, 1.95] (length/2 = 3.9/2)")
print(f"X-extent Box 2: [0.05, 3.95]")
print(f"Expected X overlap: 1.90m")
print(f"Expected Y overlap: 1.60m (full width)")
print(f"Expected intersection area: 1.90 × 1.60 = 3.04 m²")
print(f"Box area: 3.9 × 1.6 = 6.24 m²")
print(f"Expected IoU: 3.04 / (6.24 + 6.24 - 3.04) = 3.04 / 9.44 = 0.322")

print(f"\nOLD (broken) IoU: {iou_old[0, 0]:.4f}")
print(f"NEW (fixed) IoU:  {iou_new[0, 0]:.4f}")

if abs(iou_new[0, 0] - 0.322) < 0.01:
    print("✅ PASS: Fixed version gives correct IoU")
else:
    print(f"❌ FAIL: Expected ~0.322, got {iou_new[0, 0]:.4f}")

print("\n" + "="*80)
print("TEST CASE 3: Anchor vs GT (Real World Example)")
print("="*80)

# Typical scenario: anchor close to GT car
anchor = torch.tensor([[10.0, 5.0, -1.0, 1.5, 1.6, 3.9, 0.0]])  # Standard car anchor
gt_car = torch.tensor([[10.5, 5.2, -0.8, 1.6, 1.7, 4.0, 0.1]])  # Actual car (slightly different)

iou_old = bev_iou_OLD_BROKEN(anchor, gt_car)
iou_new = bev_iou_NEW_FIXED(anchor, gt_car)

print(f"Anchor:  center=(10.0, 5.0), size=(h=1.5, w=1.6, l=3.9)")
print(f"GT Car:  center=(10.5, 5.2), size=(h=1.6, w=1.7, l=4.0)")
print(f"\nOLD (broken) IoU: {iou_old[0, 0]:.4f}")
print(f"NEW (fixed) IoU:  {iou_new[0, 0]:.4f}")

if iou_new[0, 0] > 0.5:
    print(f"✅ PASS: Fixed version gives reasonable IoU (>{0.5}) for close match")
else:
    print(f"⚠️  WARNING: IoU is {iou_new[0, 0]:.4f}, may still have issues")

print("\n" + "="*80)
print("TEST CASE 4: No Overlap")
print("="*80)

# Two cars far apart
box1 = torch.tensor([[0.0, 0.0, -1.0, 1.5, 1.6, 3.9, 0.0]])
box2 = torch.tensor([[10.0, 10.0, -1.0, 1.5, 1.6, 3.9, 0.0]])

iou_old = bev_iou_OLD_BROKEN(box1, box2)
iou_new = bev_iou_NEW_FIXED(box1, box2)

print(f"Box 1: center=(0.0, 0.0)")
print(f"Box 2: center=(10.0, 10.0)")
print(f"\nOLD (broken) IoU: {iou_old[0, 0]:.4f}")
print(f"NEW (fixed) IoU:  {iou_new[0, 0]:.4f}")
print(f"Expected: 0.0000 (no overlap)")

if iou_new[0, 0] < 0.001:
    print("✅ PASS: Fixed version gives zero IoU")
else:
    print("❌ FAIL: Should be zero")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n🔍 The Bug:")
print("   The OLD code used WIDTH (index 4) for X-axis overlap calculation")
print("   But in LiDAR coordinates:")
print("   - X-axis = FORWARD direction = should use LENGTH (index 5)")
print("   - Y-axis = LEFT direction = should use WIDTH (index 4)")
print("\n✅ The Fix:")
print("   Changed line in bev_iou():")
print("   OLD: x_overlap uses w1/w2 (WRONG)")
print("   NEW: x_overlap uses l1/l2 (CORRECT)")
print("\n💡 Impact:")
print("   This bug caused:")
print("   1. Incorrect anchor-GT matching during training")
print("   2. Wrong positive/negative anchor assignment")
print("   3. Model learning from wrong supervision signals")
print("   4. Predicted boxes not aligning with GT even during overfitting")
print("\n" + "="*80)