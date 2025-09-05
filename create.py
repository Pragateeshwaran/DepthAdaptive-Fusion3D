"""
PointNet++ + Depth Attention Mechanism (DAM) multi-scale fusion
from the paper: "Multi-Modal Fusion Based on Depth Adaptive Mechanism for 3D Object Detection"

This is an educational, minimal-yet-complete PyTorch implementation that shows:
  • PointNet++ backbone with 4 Set Abstraction (SA) and 4 Feature Propagation (FP) layers
  • Image backbone that produces 4 scales to match the SA pyramid
  • Mapping & sampling: project LiDAR points to image plane and bilinear-sample image features
  • Gated weight generation (per-point, per-scale) for LiDAR & Image streams
  • Adaptive Threshold Generation Network (ATGN) from point density
  • Depth-aware fusion with near/far splitting + index-connection stitching

Notes
-----
• This code aims for clarity over speed and omits many training/infra details.
• Grouping uses naive torch.cdist-based radius grouping (O(N^2)) for readability.
• Replace with CUDA ops (e.g., from OpenPCDet/MMDetection3D) for production.
• Image backbone is a small CNN pyramid; replace with a stronger encoder if needed.
• Camera calib should be provided (intrinsics K, rectification R_rect, T_velo_to_cam).

Inputs (example shapes)
-----------------------
points:  (B, N, 3)            # xyz in LiDAR frame (Velodyne)
feats:   (B, N, C_in) or None  # optional extra per-point features; set to None to use only xyz
image:   (B, 3, H, W)          # RGB image
calib: dict with keys {"K", "R_rect", "T_velo_to_cam"} as (B, 3, 3)/(B, 4, 4)

Outputs
-------
A dict of fused feature tensors at point level. The final detection heads are not included.

Copyright
---------
This code is provided for educational purposes.
"""
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility layers
# -----------------------------
class SharedMLP(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        layers = [nn.Conv1d(in_ch, out_ch, 1, bias=not bn)]
        if bn:
            layers += [nn.BatchNorm1d(out_ch)]
        layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # (B, C_in, N)
        return self.net(x)

class SharedMLP2d(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 1, bias=not bn)]
        if bn:
            layers += [nn.BatchNorm2d(out_ch)]
        layers += [nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # (B, C_in, N, K)
        return self.net(x)

# -----------------------------
# Grouping (naive reference)
# -----------------------------
@torch.no_grad()
def radius_group(xyz: torch.Tensor, centroids: torch.Tensor, radius: float, max_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    xyz:       (B, N, 3)
    centroids: (B, M, 3)
    Return: idx (B, M, K) indices of neighbors, mask (B, M, K) [bool]
    """
    B, N, _ = xyz.shape
    M = centroids.shape[1]
    dists = torch.cdist(centroids, xyz)  # (B, M, N)
    mask = dists <= radius
    # Take top-k nearest within radius
    idx = dists.argsort(dim=-1)[:, :, :max_k]  # (B,M,K)
    # Ensure those chosen are within radius
    pick_mask = torch.gather(mask, 2, idx)
    # Fallback: if no neighbor within radius, still keep the nearest ones but zero-mask later
    return idx, pick_mask

@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, m: int) -> torch.Tensor:
    """ Naive FPS for clarity. xyz: (B, N, 3) -> idx: (B, m) """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, m, dtype=torch.long, device=device)
    # start from a random point per batch
    farthest = torch.randint(0, N, (B,), device=device)
    dist = torch.full((B, N), 1e10, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(m):
        idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B,1,3)
        d = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B,N)
        dist = torch.minimum(dist, d)
        farthest = torch.max(dist, dim=1)[1]
    return idx

# -----------------------------
# Set Abstraction (SA) and Feature Propagation (FP)
# -----------------------------
class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_out: int, radius: float, k: int, mlp_channels: Tuple[int, ...]):
        super().__init__()
        mlps = []
        last = mlp_channels[0]
        for ch in mlp_channels[1:]:
            mlps.append(SharedMLP2d(last, ch))
            last = ch
        self.mlp = nn.Sequential(*mlps)
        self.radius = radius
        self.k = k
        self.n_out = n_out

    def forward(self, xyz: torch.Tensor, feats: Optional[torch.Tensor]):
        """
        xyz:   (B, N, 3)
        feats: (B, C, N) or None
        returns: new_xyz (B, M, 3), new_feats (B, C_out, M)
        """
        B, N, _ = xyz.shape
        # 1) sample centroids
        idx = farthest_point_sample(xyz, self.n_out)  # (B, M)
        new_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,M,3)
        # 2) group neighbors
        n_idx, n_mask = radius_group(xyz, new_xyz, self.radius, self.k)  # (B,M,K)
        # 3) gather group xyz and feats
        grouped_xyz = torch.gather(xyz, 1, n_idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # (B,M,K,3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)  # (B,M,K,3)

        if feats is None:
            grouped_feats = grouped_xyz_norm.permute(0, 3, 1, 2)  # (B,3,M,K)
        else:
            # feats: (B,C,N) -> (B,C,M,K)
            grouped_feats = torch.gather(feats.transpose(1, 2), 1, n_idx.unsqueeze(-1).expand(-1, -1, -1, feats.shape[1]))
            grouped_feats = grouped_feats.permute(0, 3, 1, 2)  # (B,C,M,K)
            grouped_feats = torch.cat([grouped_feats, grouped_xyz_norm.permute(0, 3, 1, 2)], dim=1)  # add relative xyz

        # apply shared MLP + masked max pool over K
        x = self.mlp(grouped_feats)  # (B,C',M,K)
        # mask invalid neighbors
        mask = n_mask.unsqueeze(1)  # (B,1,M,K)
        x = x.masked_fill(~mask, float('-inf'))
        x = torch.max(x, dim=-1)[0]  # (B,C',M)
        return new_xyz, x

class FeaturePropagation(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.mlp = SharedMLP(in_ch, out_ch)

    def forward(self, xyz1, xyz2, feats1, feats2):
        """
        Interpolate feats from (xyz2, feats2) to (xyz1, ?), then concat with feats1 and MLP.
        xyz1: (B,N1,3) target dense;  feats1: (B,C1,N1) skip
        xyz2: (B,N2,3) source sparse; feats2: (B,C2,N2)
        returns: (B, out_ch, N1)
        """
        # inverse distance weighted interpolation (3-NN)
        dists = torch.cdist(xyz1, xyz2)  # (B,N1,N2)
        knn = torch.topk(dists, k=min(3, xyz2.shape[1]), largest=False, dim=-1)
        idx = knn.indices  # (B,N1,3)
        d = torch.gather(dists, -1, idx) + 1e-8
        w = 1.0 / d
        w = w / w.sum(dim=-1, keepdim=True)
        # gather feats2
        B, N1, K = idx.shape
        C2 = feats2.shape[1]
        gathered = torch.gather(feats2.transpose(1, 2), 1, idx.unsqueeze(-1).expand(-1, -1, -1, C2))  # (B,N1,3,C2)
        interpolated = (gathered * w.unsqueeze(-1)).sum(dim=2)  # (B,N1,C2)
        interpolated = interpolated.permute(0, 2, 1)  # (B,C2,N1)
        if feats1 is None:
            x = interpolated
        else:
            x = torch.cat([interpolated, feats1], dim=1)
        return self.mlp(x)

# -----------------------------
# Image backbone (toy pyramid)
# -----------------------------
class ImagePyramid(nn.Module):
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            )
        self.b1 = block(in_ch, base)      # 1x
        self.d1 = nn.Conv2d(base, base, 3, stride=2, padding=1)  # downsample
        self.b2 = block(base, base*2)     # 1/2
        self.d2 = nn.Conv2d(base*2, base*2, 3, stride=2, padding=1)
        self.b3 = block(base*2, base*4)   # 1/4
        self.d3 = nn.Conv2d(base*4, base*4, 3, stride=2, padding=1)
        self.b4 = block(base*4, base*8)   # 1/8

    def forward(self, x):
        # x: (B,3,H,W)
        f1 = self.b1(x)              # (B,base,H,W)
        f2 = self.b2(self.d1(f1))    # (B,2b,H/2,W/2)
        f3 = self.b3(self.d2(f2))    # (B,4b,H/4,W/4)
        f4 = self.b4(self.d3(f3))    # (B,8b,H/8,W/8)
        return [f1, f2, f3, f4]

# -----------------------------
# Mapping & bilinear sampling of image features at point locations
# -----------------------------
class MappingSampler(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pts_cam: torch.Tensor, img_feats: torch.Tensor, K: torch.Tensor):
        """
        pts_cam:  (B, N, 3) in camera frame
        img_feats: (B, C, H, W) feature map (one scale)
        K: (B, 3, 3) intrinsics
        returns: (B, C, N) sampled per-point image features
        """
        B, N, _ = pts_cam.shape
        C, H, W = img_feats.shape[1:]
        # project to pixels
        x = pts_cam[..., 0]; y = pts_cam[..., 1]; z = pts_cam[..., 2].clamp(min=1e-6)
        fx = K[:, 0, 0].unsqueeze(-1); fy = K[:, 1, 1].unsqueeze(-1)
        cx = K[:, 0, 2].unsqueeze(-1); cy = K[:, 1, 2].unsqueeze(-1)
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        # normalize to [-1,1] for grid_sample
        u_norm = 2.0 * (u / (W - 1)) - 1.0
        v_norm = 2.0 * (v / (H - 1)) - 1.0
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2)  # (B,N,1,2)
        # sample
        sampled = F.grid_sample(img_feats, grid, align_corners=True, mode='bilinear', padding_mode='zeros')  # (B,C,N,1)
        return sampled.squeeze(-1)  # (B,C,N)

# -----------------------------
# Gated Weight Generation (per paper Eqs. 2-3)
# -----------------------------
class GatedWeightGen(nn.Module):
    def __init__(self, c_img: int, c_lidar: int):
        super().__init__()
        hid = max(64, (c_img + c_lidar) // 2)
        self.fc_raw = nn.Sequential(nn.Linear(3, hid), nn.ReLU(True))          # raw xyz (FoL)
        self.fc_img = nn.Sequential(nn.Conv1d(c_img, hid, 1), nn.ReLU(True))   # FI
        self.fc_lid = nn.Sequential(nn.Conv1d(c_lidar, hid, 1), nn.ReLU(True)) # FL
        self.U = nn.Conv1d(hid, 1, 1)
        self.V = nn.Conv1d(hid, 1, 1)

    def forward(self, xyz: torch.Tensor, FI: torch.Tensor, FL: torch.Tensor):
        # xyz: (B,N,3)  FI: (B,Ci,N)  FL: (B,Cl,N)
        B, N, _ = xyz.shape
        fr = self.fc_raw(xyz.reshape(B*N, 3)).reshape(B, N, -1).permute(0,2,1)  # (B,hid,N)
        fi = self.fc_img(FI)                                                    # (B,hid,N)
        fl = self.fc_lid(FL)                                                    # (B,hid,N)
        s = torch.tanh(fr + fi + fl)                                            # (B,hid,N)
        wI = torch.sigmoid(self.U(s))                                           # (B,1,N)
        wL = torch.sigmoid(self.V(s))                                           # (B,1,N)
        FgI = FI * wI
        FgL = FL * wL
        return FgI, FgL, wI, wL

# -----------------------------
# Adaptive Threshold Generation Network (density -> threshold)
# -----------------------------
class AdaptiveThresholdNet(nn.Module):
    def __init__(self, radius: float = 1.0):
        super().__init__()
        self.radius = radius
        self.mlp = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(True),
            nn.Linear(64, 64), nn.ReLU(True),
            nn.Linear(64, 1), nn.Sigmoid()  # output in [0,1]
        )
        # Map sigmoid output to a physical depth range [min_d, max_d]
        self.min_d = 20.0
        self.max_d = 60.0

    @torch.no_grad()
    def _volume_density(self, xyz: torch.Tensor) -> torch.Tensor:
        # naive: count neighbors within radius -> density ~ count / volume
        B, N, _ = xyz.shape
        idx = radius_group(xyz, xyz, self.radius, max_k=64)[0]  # (B,N,K)
        counts = torch.ones_like(idx, dtype=torch.float).sum(dim=-1, keepdim=True)  # (B,N,1) ~ K (approx)
        # sphere volume ~ (4/3)pi r^3; use fixed radius for simplicity
        vol = (4.0/3.0) * 3.14159 * (self.radius ** 3)
        dens = counts / vol  # (B,N,1)
        return dens

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        # xyz: (B,N,3) in LiDAR frame; use Euclidean depth r = ||(x,y,z)||
        dens = self._volume_density(xyz)  # (B,N,1)
        # global pooling to a scene-level statistic
        d_mean = dens.mean(dim=1)  # (B,1)
        t = self.mlp(d_mean)       # (B,1) in [0,1]
        depth_thr = self.min_d + (self.max_d - self.min_d) * t  # (B,1)
        return depth_thr.squeeze(1)  # (B,)

# -----------------------------
# Depth-Aware Fusion module (per-scale)
# -----------------------------
class DAMFusion(nn.Module):
    def __init__(self, c_img: int, c_lidar: int):
        super().__init__()
        self.gate = GatedWeightGen(c_img, c_lidar)

    def forward(self, xyz_lidar: torch.Tensor, pts_cam: torch.Tensor,
                FL: torch.Tensor, FI: torch.Tensor, depth_thr: torch.Tensor):
        """
        xyz_lidar: (B,N,3) in LiDAR frame (for gating raw coords)
        pts_cam:   (B,N,3) in camera frame (for depth splitting using z)
        FL:        (B,Cl,N) LiDAR features
        FI:        (B,Ci,N) image per-point features
        depth_thr: (B,) per-batch threshold (meters)

        Returns: fused (B, Cl+Ci, N) with near-range using [FL || gated(FI)] and far-range using [FI || gated(FL)]
        """
        B, N, _ = xyz_lidar.shape
        FgI, FgL, wI, wL = self.gate(xyz_lidar, FI, FL)  # (B,Ci,N),(B,Cl,N)
        z = pts_cam[..., 2]  # (B,N)
        # masks
        near_mask = (z <= depth_thr.unsqueeze(1))  # (B,N)
        far_mask  = ~near_mask
        # initialize fused with zeros, then index-connect without reordering
        fused = torch.zeros(B, FL.shape[1] + FI.shape[1], N, device=FL.device)
        # near: concat [FL, FgI]
        near = torch.cat([FL, FgI], dim=1)
        # far:  concat [FI, FgL]
        far = torch.cat([FI, FgL], dim=1)
        # scatter via masks (index-connection)
        fused[:, :, near_mask] = near[:, :, near_mask]
        fused[:, :, far_mask]  = far[:, :, far_mask]
        return fused, (wI, wL)

# -----------------------------
# Full Multi-Scale Architecture
# -----------------------------
class PointNetPP_DAM(nn.Module):
    def __init__(self, k_group=32):
        super().__init__()
        # SA pyramid
        self.sa1 = PointNetSetAbstraction(4096, radius=0.8, k=k_group, mlp_channels=(3, 64, 64, 128))
        self.sa2 = PointNetSetAbstraction(1024, radius=1.2, k=k_group, mlp_channels=(128+3, 128, 128, 256))
        self.sa3 = PointNetSetAbstraction(256,  radius=2.4, k=k_group, mlp_channels=(256+3, 256, 256, 512))
        self.sa4 = PointNetSetAbstraction(64,   radius=4.8, k=k_group, mlp_channels=(512+3, 512, 512, 1024))
        # FP
        self.fp4 = FeaturePropagation(1024+512, 512)
        self.fp3 = FeaturePropagation(512+256,  256)
        self.fp2 = FeaturePropagation(256+128,  128)
        self.fp1 = FeaturePropagation(128,       128)
        # Image encoder
        self.img_enc = ImagePyramid(in_ch=3, base=32)
        # Per-scale DAM fusion blocks (match channels)
        self.dam1 = DAMFusion(c_img=32,   c_lidar=128)
        self.dam2 = DAMFusion(c_img=64,   c_lidar=256)
        self.dam3 = DAMFusion(c_img=128,  c_lidar=512)
        self.dam4 = DAMFusion(c_img=256,  c_lidar=1024)
        # mapping sampler
        self.sampler = MappingSampler()
        # adaptive threshold net
        self.atg = AdaptiveThresholdNet(radius=1.0)

    def lidar_to_cam(self, pts_lidar: torch.Tensor, R_rect: torch.Tensor, T_velo_to_cam: torch.Tensor):
        """Transform LiDAR xyz (Velodyne) to camera frame using calibration matrices.
        pts_lidar: (B,N,3); R_rect: (B,3,3); T_velo_to_cam: (B,4,4)
        """
        B, N, _ = pts_lidar.shape
        ones = torch.ones(B, N, 1, device=pts_lidar.device)
        pts_h = torch.cat([pts_lidar, ones], dim=-1).transpose(1, 2)  # (B,4,N)
        pts_cam = (T_velo_to_cam @ pts_h)  # (B,4,N)
        pts_cam = pts_cam[:, :3, :]  # (B,3,N)
        pts_cam = (R_rect @ pts_cam).transpose(1, 2)  # (B,N,3)
        return pts_cam

    def forward(self, points: torch.Tensor, image: torch.Tensor, calib: Dict[str, torch.Tensor], feats: Optional[torch.Tensor]=None):
        """
        points: (B,N,3) LiDAR xyz
        image:  (B,3,H,W)
        calib:  dict with keys K (B,3,3), R_rect (B,3,3), T_velo_to_cam (B,4,4)
        feats:  (B,N,C_in) or None
        Returns dict with fused per-point features (final scale) and intermediates.
        """
        B, N, _ = points.shape
        K, R_rect, T_v2c = calib["K"], calib["R_rect"], calib["T_velo_to_cam"]

        # Image features pyramid
        f1, f2, f3, f4 = self.img_enc(image)  # (B,32,H,W), (B,64,H/2,W/2), ...

        # SA hierarchy
        xyz1, f_l1 = self.sa1(points, feats)               # (B,4096,3), (B,128,4096)
        xyz2, f_l2 = self.sa2(xyz1, torch.cat([f_l1, xyz1.transpose(1,2)], dim=1))  # (B,1024,3),(B,256,1024)
        xyz3, f_l3 = self.sa3(xyz2, torch.cat([f_l2, xyz2.transpose(1,2)], dim=1))  # (B,256,3), (B,512,256)
        xyz4, f_l4 = self.sa4(xyz3, torch.cat([f_l3, xyz3.transpose(1,2)], dim=1))  # (B,64,3),  (B,1024,64)

        # Project LiDAR points at each scale to camera and sample image feats
        pts_cam1 = self.lidar_to_cam(xyz1, R_rect, T_v2c)  # (B,4096,3)
        pts_cam2 = self.lidar_to_cam(xyz2, R_rect, T_v2c)
        pts_cam3 = self.lidar_to_cam(xyz3, R_rect, T_v2c)
        pts_cam4 = self.lidar_to_cam(xyz4, R_rect, T_v2c)

        # Downsample image features to matching resolutions for sampling
        # We use the closest pyramid scale: xyz1<-f2, xyz2<-f3, xyz3<-f4, xyz4<-f4 (more stride). Adjust as desired.
        FI1 = self.sampler(pts_cam1, f2, K)  # (B,64,4096)
        FI2 = self.sampler(pts_cam2, f3, K)  # (B,128,1024)
        FI3 = self.sampler(pts_cam3, f4, K)  # (B,256,256)
        FI4 = self.sampler(pts_cam4, f4, K)  # (B,256,64)

        # Adaptive threshold (scene-level)
        depth_thr = self.atg(points)  # (B,)

        # DAM fusion per scale
        fused4, _ = self.dam4(xyz4, pts_cam4, f_l4, FI4, depth_thr)  # (B,1024+256,64)
        # Propagate up
        f_up3 = self.fp4(xyz3, xyz4, f_l3, fused4)  # (B,512,256)
        fused3, _ = self.dam3(xyz3, pts_cam3, f_up3, FI3, depth_thr)  # (B,512+128,256) -> channels 640; simplify by projecting to 512 in FP
        # To keep channels consistent, reduce after fusion for next FP
        f3_red = SharedMLP(fused3.shape[1], 512)(fused3)  # (B,512,256)

        f_up2 = self.fp3(xyz2, xyz3, f_l2, f3_red)  # (B,256,1024)
        fused2, _ = self.dam2(xyz2, pts_cam2, f_up2, FI2, depth_thr)  # (B,256+64,1024)
        f2_red = SharedMLP(fused2.shape[1], 256)(fused2)

        f_up1 = self.fp2(xyz1, xyz2, f_l1, f2_red)  # (B,128,4096)
        fused1, _ = self.dam1(xyz1, pts_cam1, f_up1, FI1, depth_thr)  # (B,128+32,4096)
        f1_red = SharedMLP(fused1.shape[1], 128)(fused1)

        final = self.fp1(points, xyz1, None, f1_red)  # (B,128,N)

        return {
            "fused_final": final,      # (B,128,N) per-point fused features
            "depth_thr": depth_thr,    # (B,)
            "xyz": points,             # (B,N,3)
            "multi": {
                "xyz1": xyz1, "xyz2": xyz2, "xyz3": xyz3, "xyz4": xyz4,
                "f_l1": f_l1, "f_l2": f_l2, "f_l3": f_l3, "f_l4": f_l4,
            }
        }


# -----------------------------
# Quick smoke test
# -----------------------------
if __name__ == "__main__":
    B, N, H, W = 2, 16384, 384, 1280
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNetPP_DAM().to(device)
    pts = (torch.rand(B, N, 3, device=device) - 0.5) * torch.tensor([80.0, 4.0, 70.4], device=device)  # roughly KITTI range
    img = torch.rand(B, 3, H, W, device=device)
    # simple calib (identity-ish) for demo; replace with real KITTI calib
    K = torch.tensor([[[1000.0, 0.0, W/2], [0.0, 1000.0, H/2], [0.0, 0.0, 1.0]]], device=device).repeat(B,1,1)
    R_rect = torch.eye(3, device=device).unsqueeze(0).repeat(B,1,1)
    T_v2c = torch.eye(4, device=device).unsqueeze(0).repeat(B,1,1)
    out = model(pts, img, {"K": K, "R_rect": R_rect, "T_velo_to_cam": T_v2c}, feats=None)
    print(out["fused_final"].shape, out["depth_thr"])
