# DepthAdaptive-Fusion3D

### Folder

src/
│   __init__.py           # makes src a package
│   train.py              # training entry point
│   evaluate.py           # evaluation entry point
│   inference.py          # run inference on new data
│
├───data
│   │   __init__.py
│   │   dataloader.py     # loads KITTI data (images, lidar, labels)
│   │   preprocess.py     # calibration, projection, normalization
│
├───models
│   │   __init__.py
│   │   backbone.py       # PointNet++ / CNN backbone
│   │   fusion.py         # Depth Attention Mechanism (DAM)
│   │   threshold_net.py  # Adaptive Threshold Generation Network
│   │   detection_head.py # final 3D bounding box regression/classification
│
├───utils
│   │   __init__.py
│   │   visualization.py  # 3D/2D bounding box plotting
│   │   metrics.py        # mAP, IoU, KITTI evaluation scripts
│   │   config.py         # hyperparameters and paths
│
└───experiments
    │   __init__.py
    │   ablation.py       # run ablation studies
    │   analysis.py       # analyze results, generate plots
