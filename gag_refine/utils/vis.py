import open3d as o3d
import burg_toolkit as burg


def sphere(point):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    s.translate(point, relative=True)
    s.paint_uniform_color([0, 0, 0])
    s.compute_triangle_normals()
    return s


def spheres(points):
    return [sphere(point) for point in points]


def show_frame_axes(pose, origin=None, show=True):
    vis_objs = []
    for i in range(3):
        axis = pose[:3, i]

        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.003, cylinder_height=0.02,
                                                       cone_height=0.01)
        rot = burg.util.rotation_to_align_vectors([0, 0, 1], axis)
        arrow.rotate(rot, [0, 0, 0])
        color = [0]*3
        color[i] = 1
        arrow.paint_uniform_color(color)
        arrow.compute_triangle_normals()
        if origin is not None:
            arrow.translate(origin, relative=True)
        vis_objs.append(arrow)

    if origin is not None:
        vis_objs.append(sphere(origin))

    if show:
        burg.visualization.show_geometries(vis_objs)
    return vis_objs
