import os
import torch
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import data_loader
from GNN import intializer

ROOT = r'F:\Work\DeepLearning\Research\V2X-Radar-V'
OUT  = r'F:\Work\DeepLearning\Research\graphs'
os.makedirs(OUT, exist_ok=True)

lidar_data = data_loader.get_LiDAR(ROOT, split='training')

for i, points in enumerate(lidar_data):
    graph = intializer.construct_graph(
        points,
        radius=4.0,
        num_vertices=512,
        verbose=False
    )
    torch.save(graph, f"{OUT}/{i}.pt")

print("Graph precomputation done")
