import os
import open3d as o3d
import burg_toolkit as burg
burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.IPYNB_VIEWER)

scene_idx = '0245'  # choose one from [0050, 0051, 0052, 0053, 0054]
out_path = 'out/gag_3grid_fullpc_with_partialpc_sdf/generation/'

ply_input = os.path.join(out_path, f'input/scenes/{scene_idx}.ply')
off_output = os.path.join(out_path, f'meshes/scenes/{scene_idx}.off')

input_pc = o3d.io.read_point_cloud(ply_input)
output_mesh = o3d.io.read_triangle_mesh(off_output)

burg.visualization.show_geometries([input_pc])
burg.visualization.show_geometries([output_mesh])