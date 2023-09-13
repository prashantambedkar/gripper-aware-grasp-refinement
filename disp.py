import os
import numpy as np
import open3d as o3d
from gag_refine.dataset import load_object_library
import burg_toolkit as burg
burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.IPYNB_VIEWER)

DATA_DIR = 'data/gag-refine/scenes/'
lib = load_object_library(os.path.join(DATA_DIR, '..'))
print(f'loaded object library with {len(lib)} objects')

scene_idx = '0245'  # can go from 0000 up to 0054
scene, _, _ = burg.Scene.from_yaml(os.path.join(DATA_DIR, f'{scene_idx}/scene.yaml'), lib)
print(scene)

burg.visualization.show_geometries([scene])