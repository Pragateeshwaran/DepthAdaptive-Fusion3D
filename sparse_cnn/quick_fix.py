"""
SIMPLE CHECK: What are the model's classification scores?
"""
import torch
import sys
sys.path.append(r'F:\Work\DeepLearning\Research\sparse_cnn')

# Just load the checkpoint and check the score threshold
CHECKPOINT_PATH = r'F:\Work\DeepLearning\Research\checkpoint_epoch_40.pth'

print("="*80)
print("SIMPLE DIAGNOSTIC")
print("="*80)

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

print(f"\nCheckpoint Info:")
print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"  Loss: {checkpoint.get('loss', 'unknown')}")

if 'loss_stats' in checkpoint:
    print(f"\nLoss Statistics:")
    for key, value in checkpoint['loss_stats'].items():
        print(f"  {key}: {value}")
    
    num_pos = checkpoint['loss_stats'].get('num_pos_anchors', 0)
    print(f"\n{'='*80}")
    if num_pos > 0:
        print(f"✅ Model was trained with {num_pos:.1f} positive anchors per batch")
        print(f"   This means anchors WERE matching GT boxes during training")
    else:
        print(f"❌ Model was trained with ZERO positive anchors!")
        print(f"   This means the model never learned anything useful")
        print(f"   SOLUTION: Need to retrain from scratch with correct anchors")
    print(f"{'='*80}")
else:
    print("\n⚠️  No loss_stats in checkpoint")

print("\n" + "="*80)
print("NEXT STEP:")
print("="*80)
print("Edit rpn_refinement.py:")
print("  Line ~266: Change score_thresh from 0.01 to 0.001")
print("  This will lower the threshold to see ANY predictions")
print("\nThen run: python visualize_predictions.py")
print("="*80)