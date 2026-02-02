"""
Quick Fix Script - Apply Immediate Improvements

This script patches your existing files with immediate fixes for:
1. Overlapping boxes (stricter NMS)
2. Better training configuration

Run this before continuing training!
"""

import os
import shutil
from datetime import datetime


def backup_file(filepath):
    """Create timestamped backup of a file."""
    if not os.path.exists(filepath):
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}.backup_{timestamp}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def patch_rpn_refinement(filepath="rpn_refinement.py"):
    """
    Patch rpn_refinement.py with stricter NMS parameters.
    
    Changes:
    - NMS threshold: 0.5 → 0.3
    - Score threshold: 0.05 → 0.2
    - Reduce number of proposals
    """
    print(f"\n{'='*70}")
    print(f"PATCHING: {filepath}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Backup
    backup = backup_file(filepath)
    print(f"✓ Backup created: {backup}")
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply patches
    patches_applied = 0
    
    # Patch 1: ProposalGenerator NMS threshold
    if "nms_thresh=0.5" in content:
        content = content.replace(
            "nms_thresh=0.5",
            "nms_thresh=0.3  # ✓ PATCHED: Stricter NMS"
        )
        patches_applied += 1
        print("✓ Patch 1: NMS threshold 0.5 → 0.3")
    
    # Patch 2: ProposalGenerator score threshold
    if "score_thresh=0.05" in content:
        content = content.replace(
            "score_thresh=0.05",
            "score_thresh=0.2  # ✓ PATCHED: Higher confidence"
        )
        patches_applied += 1
        print("✓ Patch 2: Score threshold 0.05 → 0.2")
    
    # Patch 3: Pre-NMS top N (training)
    if "pre_nms_top_n_train=2000" in content:
        content = content.replace(
            "pre_nms_top_n_train=2000",
            "pre_nms_top_n_train=1000  # ✓ PATCHED: Fewer proposals"
        )
        patches_applied += 1
        print("✓ Patch 3: Pre-NMS train 2000 → 1000")
    
    # Patch 4: Pre-NMS top N (test)
    if "pre_nms_top_n_test=1000" in content:
        content = content.replace(
            "pre_nms_top_n_test=1000",
            "pre_nms_top_n_test=500  # ✓ PATCHED: Fewer proposals"
        )
        patches_applied += 1
        print("✓ Patch 4: Pre-NMS test 1000 → 500")
    
    # Patch 5: Post-NMS top N (training)
    if "post_nms_top_n_train=500" in content:
        content = content.replace(
            "post_nms_top_n_train=500",
            "post_nms_top_n_train=300  # ✓ PATCHED: Stricter filtering"
        )
        patches_applied += 1
        print("✓ Patch 5: Post-NMS train 500 → 300")
    
    # Patch 6: Post-NMS top N (test)
    if "post_nms_top_n_test=100" in content:
        content = content.replace(
            "post_nms_top_n_test=100",
            "post_nms_top_n_test=50  # ✓ PATCHED: Stricter filtering"
        )
        patches_applied += 1
        print("✓ Patch 6: Post-NMS test 100 → 50")
    
    # Write patched file
    if patches_applied > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Applied {patches_applied} patches to {filepath}")
        return True
    else:
        print(f"\n⚠️  No patches applied (file may already be patched)")
        return True


def patch_trail(filepath="trail.py"):
    """
    Patch trail.py with improved training configuration.
    
    Changes:
    - Epochs: 50 → 100
    - Learning rate: 5e-4 → 1e-4
    - Samples: 100 → 500
    """
    print(f"\n{'='*70}")
    print(f"PATCHING: {filepath}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Backup
    backup = backup_file(filepath)
    print(f"✓ Backup created: {backup}")
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply patches
    patches_applied = 0
    
    # Patch 1: Increase epochs
    if "NUM_EPOCHS = 50" in content:
        content = content.replace(
            "NUM_EPOCHS = 50",
            "NUM_EPOCHS = 100  # ✓ PATCHED: Train longer"
        )
        patches_applied += 1
        print("✓ Patch 1: Epochs 50 → 100")
    
    # Patch 2: Lower learning rate
    if "LEARNING_RATE = 5e-4" in content:
        content = content.replace(
            "LEARNING_RATE = 5e-4",
            "LEARNING_RATE = 1e-4  # ✓ PATCHED: More stable learning"
        )
        patches_applied += 1
        print("✓ Patch 2: LR 5e-4 → 1e-4")
    
    # Patch 3: More training samples
    if "NUM_SAMPLES = 100" in content:
        content = content.replace(
            "NUM_SAMPLES = 100",
            "NUM_SAMPLES = 500  # ✓ PATCHED: More data"
        )
        patches_applied += 1
        print("✓ Patch 3: Samples 100 → 500")
    
    # Patch 4: Better optimizer (Adam → AdamW)
    if "optim.Adam(model.parameters()" in content and "optim.AdamW" not in content:
        content = content.replace(
            "optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)",
            "optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # ✓ PATCHED: Better optimizer"
        )
        patches_applied += 1
        print("✓ Patch 4: Adam → AdamW with weight decay")
    
    # Write patched file
    if patches_applied > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Applied {patches_applied} patches to {filepath}")
        return True
    else:
        print(f"\n⚠️  No patches applied (file may already be patched)")
        return True


def patch_visualize_predictions(filepath="visualize_predictions.py"):
    """
    Patch visualize_predictions.py with stricter visualization parameters.
    
    Changes:
    - NMS threshold: 0.5 → 0.3
    - Score threshold: 0.1 → 0.3
    """
    print(f"\n{'='*70}")
    print(f"PATCHING: {filepath}")
    print(f"{'='*70}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Backup
    backup = backup_file(filepath)
    print(f"✓ Backup created: {backup}")
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply patches
    patches_applied = 0
    
    # Patch 1: NMS threshold in function call
    if "nms_threshold=0.5" in content:
        content = content.replace(
            "nms_threshold=0.5",
            "nms_threshold=0.3  # ✓ PATCHED: Stricter NMS"
        )
        patches_applied += 1
        print("✓ Patch 1: Visualization NMS 0.5 → 0.3")
    
    # Patch 2: Score threshold
    if "score_threshold=0.1" in content:
        content = content.replace(
            "score_threshold=0.1",
            "score_threshold=0.3  # ✓ PATCHED: Higher confidence"
        )
        patches_applied += 1
        print("✓ Patch 2: Visualization score 0.1 → 0.3")
    
    # Write patched file
    if patches_applied > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✅ Applied {patches_applied} patches to {filepath}")
        return True
    else:
        print(f"\n⚠️  No patches applied (file may already be patched)")
        return True


def main():
    """Apply all patches."""
    print("\n" + "="*70)
    print("QUICK FIX SCRIPT - IMMEDIATE IMPROVEMENTS")
    print("="*70)
    print("\nThis script will:")
    print("  1. ✓ Fix overlapping boxes (stricter NMS)")
    print("  2. ✓ Improve training configuration")
    print("  3. ✓ Create backups of all modified files")
    print("\n" + "="*70)
    
    input("\nPress Enter to continue (or Ctrl+C to cancel)...")
    
    # Patch files
    results = {
        'rpn_refinement.py': patch_rpn_refinement(),
        'trail.py': patch_trail(),
        'visualize_predictions.py': patch_visualize_predictions(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("PATCH SUMMARY")
    print("="*70)
    
    for filename, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {filename}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n" + "="*70)
        print("✅ ALL PATCHES APPLIED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Run trail.py to continue training with better config")
        print("  2. Training will resume from checkpoint_epoch_4.pth")
        print("  3. Visualizations will have fewer overlapping boxes")
        print("\nTo restore original files:")
        print("  - Look for .backup_* files in the same directory")
        print("  - Copy them back to original filenames")
        print("\n" + "="*70)
    else:
        print("\n" + "="*70)
        print("⚠️  SOME PATCHES FAILED")
        print("="*70)
        print("\nPlease check:")
        print("  - File paths are correct")
        print("  - You have write permissions")
        print("  - Files exist in current directory")
    
    return all_success


if __name__ == "__main__":
    import sys
    
    # Check if running from correct directory
    if not os.path.exists("trail.py") and not os.path.exists("rpn_refinement.py"):
        print("\n❌ ERROR: Cannot find trail.py or rpn_refinement.py")
        print("Please run this script from the directory containing your code files.")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1)