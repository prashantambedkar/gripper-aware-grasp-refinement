import os
import datetime
import time

import numpy as np
import burg_toolkit as burg
import open3d as o3d
import matplotlib.pyplot as plt

from gag_refine.dataset.ferrari_canny import FerrariCanny
from gag_refine.dataset import load_object_library
from convonets.src.utils.libmesh import check_mesh_contains

import matplotlib
matplotlib.use('TkAgg')  # this avoids an incompatibility issue I had with OpenCV and Qt
# also see https://stackoverflow.com/a/49887744


def get_grasp_visualisation(contact_points, contact_normals, with_cones=False):
    # contact points as spheres, normals as arrows or lines, optionally friction cones as well
    vis_objs = []
    for c, n in zip(contact_points, contact_normals):
        # visualise contact point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(c)
        sphere.paint_uniform_color([1, 0, 0])
        sphere.compute_triangle_normals()
        vis_objs.append(sphere)

        # visualise normal
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.001, cone_radius=0.003, cylinder_height=0.02,
                                                       cone_height=0.01)
        arrow.translate([0, 0, -0.03], relative=True)  # compensate for cone and cylinder height
        rot = burg.util.rotation_to_align_vectors([0, 0, 1], n)
        arrow.rotate(rot, [0, 0, 0])
        arrow.translate(c, relative=True)
        arrow.paint_uniform_color([0, 1, 0])
        arrow.compute_triangle_normals()
        vis_objs.append(arrow)

        if with_cones:
            cone = get_friction_cone(c, n)
            cone.paint_uniform_color([0.4, 0.4, 0.4])
            cone.compute_triangle_normals()
            vis_objs.append(cone)

    return vis_objs


def sample_object_surface(mesh, points_per_square_cm=20):
    """ randomly samples points from the trimesh surface. the density only affects the overall number of points.

    Returns:
        tuple of points (n, 3) and inward-facing normals (n, 3). """
    n_sample_points = int(mesh.area * points_per_square_cm * 100 * 100)  # from m^2 to cm^2
    points, face_idx = mesh.sample(n_sample_points, return_index=True)
    normals = mesh.face_normals[face_idx]
    normals = -normals / np.linalg.norm(normals, axis=-1)[..., None]  # inward facing! also normalise

    return points, normals


def get_friction_cone(point, normal, mu=0.5, n_edges=8):
    """
    creates an open3d friction cone mesh at the given point and oriented using the inward-facing(!) normal
    """
    height = 0.3  # based on max scene extents
    radius = mu * height
    cone_template = o3d.geometry.TriangleMesh.create_cone(radius=radius, height=height, resolution=n_edges)

    # move to desired pose
    cone_template.translate([0, 0, -height], relative=True)
    rot = burg.util.rotation_to_align_vectors([0, 0, -1], normal)
    cone_template.rotate(rot, [0, 0, 0])
    cone_template.translate(point, relative=True)

    return cone_template


def sample_two_finger_antipodal(points, normals, mu=0.5, n=1, skip_antipodal_check=False,
                                max_grasps_per_contact=5, visualise=False):
    """ gets points on an object surface along with their inward-facing normals and samples antipodal grasps.

    Params:
        points: (N, 3) numpy array points from the object surface
        normals: (N, 3) numpy array with inward-facing normals
        mu: friction coefficient
        n: number of grasps to sample
        skip_antipodal_check (bool): if True, skips checks for normal orientation of second contact point
        max_grasps_per_contact (int): max number of 2nd contact points for the same 1st contact point
        visualise (bool): Whether to show some nice visualisations
    Returns:
        tuple of (contact_points, contact_normals), each with shape (n, 2, 3)
    """
    contact_points = np.empty((n, 2, 3))
    contact_normals = np.empty((n, 2, 3))

    # create random order of points, then go through this order using them as first contact point
    points_order = np.arange(len(points))
    np.random.shuffle(points_order)
    points_idx = 0
    n_sampled_grasps = 0
    while n_sampled_grasps < n:
        # pick first contact point
        p1 = points[points_order[points_idx]]
        n1 = normals[points_order[points_idx]]

        # create candidate points (all except p1)
        candidate_pts = np.delete(points, points_order[points_idx], axis=0)
        candidate_normals = np.delete(normals, points_order[points_idx], axis=0)
        points_idx += 1
        if points_idx == len(points):
            points_idx = 0

        # find points inside the friction cone with our occupancy tool (we also exclude p1)
        cone = get_friction_cone(p1, n1, mu=mu)
        inside_idx = check_mesh_contains(burg.mesh_processing.as_trimesh(cone), candidate_pts)
        candidate_pts = candidate_pts[inside_idx]
        candidate_normals = candidate_normals[inside_idx]

        if len(candidate_pts) == 0:
            continue

        if skip_antipodal_check:
            target_points = candidate_pts
            target_normals = candidate_normals
        else:
            # compute angles to check antipodal constraints
            d = (candidate_pts - p1).reshape(-1, 3)
            signs = np.zeros(len(d))
            angles = burg.util.angle(d, candidate_normals, sign_array=signs, as_degree=False)

            # we retain target points where the angle is within the friction cone
            # angle must be small and direction of normal must be opposing
            # negative sign means the normals are facing towards p1, i.e. opposing
            mask_angle = np.abs(angles) <= np.arctan(mu)
            mask_direction = signs < 0
            target_points = candidate_pts[mask_angle & mask_direction]
            target_normals = candidate_normals[mask_angle & mask_direction]

        if visualise:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.translate(p1)
            sphere.compute_triangle_normals()
            cone.compute_triangle_normals()
            if len(target_points) > 1:
                burg.visualization.show_geometries([points, cone, sphere, candidate_pts, target_points])
            else:
                burg.visualization.show_geometries([points, cone, sphere, candidate_pts])

        m = len(target_points)
        # no targets found
        if m == 0:
            continue

        n_grasps_to_sample = min(max_grasps_per_contact, n-n_sampled_grasps)
        if m > n_grasps_to_sample:
            indices = np.random.choice(len(target_points), n_grasps_to_sample, replace=False)
            target_points = target_points[indices]
            target_normals = target_normals[indices]
            m = n_grasps_to_sample

        assert len(target_points.shape) == 2  # must be (-1, 3)

        # reshape contacts to (m, 2, 3), and normals as well
        first_contacts = np.repeat(p1.reshape(1, 3), m, axis=0)
        first_normals = np.repeat(n1.reshape(1, 3), m, axis=0)
        m_contacts = np.concatenate([first_contacts[:, None, :], target_points[:, None, :]], axis=1)
        m_normals = np.concatenate([first_normals[:, None, :], target_normals[:, None, :]], axis=1)

        contact_points[n_sampled_grasps:n_sampled_grasps+m] = m_contacts
        contact_normals[n_sampled_grasps:n_sampled_grasps+m] = m_normals
        n_sampled_grasps += m

        if visualise:
            for k in range(m):
                vis_objs = get_grasp_visualisation(m_contacts[k], m_normals[k], with_cones=True)
                burg.visualization.show_geometries([points, *vis_objs])

    return contact_points, contact_normals


def check_antipodal_grasp_sampling():
    # unused, just some testing code
    base_dir = 'data/gag/grasps/antipodal'
    burg.io.make_sure_directory_exists(base_dir)
    splits = ['train', 'test']
    visualise = False

    sample_pts_per_square_cm = 10
    grasps_to_sample = [(2, 500), (3, 5000), (4, 5000), (5, 5000)]

    t0 = time.time()
    for split in splits:
        lib = load_object_library(os.path.join(base_dir, '../../'), split)

        i = 1
        for name, obj_type in lib.items():
            t = datetime.datetime.now()
            print(f'{i}/{len(lib)} ({split}): obj {name}; time: {time.time()-t0:.2f}s; {t.hour:02d}:{t.minute:02d}')
            i += 1

            save_dir = os.path.join(base_dir, name)
            burg.io.make_sure_directory_exists(save_dir)

            mesh = obj_type.trimesh
            n_sample_points = int(mesh.area * sample_pts_per_square_cm * 100 * 100)  # from m^2 to cm^2

            points, face_idx = mesh.sample(n_sample_points, return_index=True)
            normals = mesh.face_normals[face_idx]
            normals = -normals / np.linalg.norm(normals, axis=-1)[..., None]  # inward facing! also normalise
            com = mesh.center_mass

            if visualise:
                print(f'area: {mesh.area}, sampled {n_sample_points} points')
                burg.visualization.show_geometries([obj_type, points])

            # two-finger grasps first
            n_contacts, n_grasps = grasps_to_sample.pop(0)
            assert n_contacts == 2, 'grasps_to_sample should start with 2 contacts'

            print(f'sampling two-finger grasps...')
            contacts_2f, normals_2f = sample_two_finger_antipodal(points, normals, n=n_grasps, mu=0.45)

            print(f'computing scores...')
            scores_dict = {}
            for method in ['full', 'alternating', 'elementary']:
                print(f'method: {method}')
                t0 = time.time()
                # for mu in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                for gamma in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
                    scores = FerrariCanny.compute_L1_scores_parallel(
                        contacts_2f, normals_2f, com, soft_contact_method=method, mu=0.5, gamma=gamma
                    )
                    scores_dict[method] = scores
                    print(f'gamma={gamma:.3f}. {np.count_nonzero(scores)}/{n_grasps}: '
                          f'best: {np.max(scores):.5f}, worst: {np.min(scores):.5f}, '
                          f'mean: {np.mean(scores):.5f}, median: {np.median(scores):.5f}')
                    # bad_scores = scores == 0
                    # for i in range(np.count_nonzero(bad_scores)):
                    #     vis_objs = get_grasp_visualisation(contacts_2f[bad_scores][i], normals_2f[bad_scores][i],
                    #                                        with_cones=True)
                    #     burg.visualization.show_geometries([points, *vis_objs])
                t1 = time.time()
                print(f'that required {t1-t0:.03f} seconds')


def sample_random_grasps(points, normals, n_contacts, n_grasps):
    """ just randomly pick contact point tuples from the object surface """
    idcs = np.random.choice(len(points), (n_grasps, n_contacts))
    contacts = points[idcs]
    contact_normals = normals[idcs]
    return contacts, contact_normals


def make_grasp_dict(contacts, normals, scores):
    grasp_dict = {
        'contact_points': contacts.astype(np.float16),
        'contact_normals': normals.astype(np.float16),
        'scores': scores
    }
    return grasp_dict


def save_grasp_dicts(directory, grasp_dicts):
    # might be a list of grasp dicts or a single grasp dict
    burg.io.make_sure_directory_exists(directory)
    if not isinstance(grasp_dicts, list):
        grasp_dicts = [grasp_dicts]

    for i, grasp_dict in enumerate(grasp_dicts):
        fn = os.path.join(directory, f'{i}.npz')
        np.savez(fn, **grasp_dict)
        print(f'saved {fn}')


def create_two_finger_grasp_annotations(base_dir):
    splits = ['train', 'test']

    t0 = time.time()
    for split in splits:
        lib = load_object_library(os.path.join(base_dir, '../'), split)

        for i, (name, obj_type) in enumerate(lib.items()):
            t = datetime.datetime.now()
            print(f'{i}/{len(lib)} ({split}): obj {name}; time: {time.time()-t0:.2f}s; {t.hour:02d}:{t.minute:02d}')
            i += 1

            save_dir = os.path.join(base_dir, name, '2')
            burg.io.make_sure_directory_exists(save_dir)

            mesh = obj_type.trimesh
            points, normals = sample_object_surface(mesh)
            com = mesh.center_mass

            # sample
            # 5000x mu=0.5, only grasps that satisfy FC
            # 5000x mu=1.0, no checks on 2nd contact → might satisfy FC or not
            # 10000x randomly sampled contact points

            all_grasps = []
            t1 = time.time()
            grasps = sample_two_finger_antipodal(points, normals, mu=0.5, n=5000, max_grasps_per_contact=5)
            scores = FerrariCanny.compute_L1_scores_parallel(grasps[0], grasps[1], com)
            all_grasps.append(make_grasp_dict(grasps[0], grasps[1], scores))
            print(f'\tcreated 5000x mu=0.5 grasps that satisfy antipodal constraint. '
                  f'successful: {np.mean(scores > 0):.3f}. ({time.time()-t1:.2f}s)')

            t1 = time.time()
            grasps = sample_two_finger_antipodal(points, normals, mu=1.0, n=5000, max_grasps_per_contact=5,
                                                 skip_antipodal_check=True)
            scores = FerrariCanny.compute_L1_scores_parallel(grasps[0], grasps[1], com)
            all_grasps.append(make_grasp_dict(grasps[0], grasps[1], scores))
            print(f'\tcreated 5000x mu=1.0 grasps without checking antipodal constraint. '
                  f'successful: {np.mean(scores > 0):.3f}. ({time.time()-t1:.2f}s)')

            t1 = time.time()
            grasps = sample_random_grasps(points, normals, n_grasps=10000, n_contacts=2)
            scores = FerrariCanny.compute_L1_scores_parallel(grasps[0], grasps[1], com)
            all_grasps.append(make_grasp_dict(grasps[0], grasps[1], scores))
            print(f'\tcreated 10000x random grasps. '
                  f'successful: {np.mean(scores > 0):.3f}. ({time.time()-t1:.2f}s)')

            save_grasp_dicts(save_dir, all_grasps)


def sample_random_grasp_annotations():
    base_dir = 'data/gag/grasps/random'
    splits = ['train', 'test']
    visualise = False

    grasps_to_sample = [(2, 10000), (3, 10000), (4, 10000), (5, 10000)]

    t0 = time.time()
    for split in splits:
        lib = load_object_library(os.path.join(base_dir, '../../'), split)

        i = 1
        for name, obj_type in lib.items():
            t = datetime.datetime.now()
            print(f'{i}/{len(lib)} ({split}): obj {name}; time: {time.time()-t0:.2f}s; {t.hour:02d}:{t.minute:02d}')
            i += 1

            save_dir = os.path.join(base_dir, name)
            burg.io.make_sure_directory_exists(save_dir)

            mesh = obj_type.trimesh
            points, normals = sample_object_surface(mesh)
            com = mesh.center_mass

            if visualise:
                print(f'area: {mesh.area}, sampled {len(points)} points')
                burg.visualization.show_geometries([obj_type, points])

            for n_contacts, n_grasps in grasps_to_sample:
                # we might sample the same point multiple times in a single grasp, but probability is very low
                idcs = np.random.choice(len(points), (n_grasps, n_contacts))

                contacts = points[idcs]
                contact_normals = normals[idcs]
                scores = FerrariCanny.compute_L1_scores_parallel(contacts, contact_normals, com)
                print(f'{np.count_nonzero(scores)}/{n_grasps} {n_contacts}-finger grasps satisfy force closure.'
                      f' best: {np.max(scores)}')
                if visualise:
                    best_idx = np.argmax(scores)
                    grasp_vis = get_grasp_visualisation(contacts[best_idx], contact_normals[best_idx])
                    burg.visualization.show_geometries([obj_type, *grasp_vis])

                grasp_dict = {
                    'contact_points': contacts,
                    'contact_normals': contact_normals,
                    'scores': scores
                }
                np.savez(os.path.join(save_dir, f'{n_contacts}.npz'), **grasp_dict)
                print(f'saved {save_dir}/{n_contacts}.npz')


def plot_score_distribution(base_dir, n_fingers=2):
    objects = sorted(os.listdir(base_dir))
    for obj in objects:
        print(f'object: {obj}')
        scores = []

        grasp_dir = os.path.join(base_dir, obj, f'{n_fingers}')
        grasp_files = os.listdir(grasp_dir)
        for fn in grasp_files:
            full_fn = os.path.join(grasp_dir, fn)
            s = np.load(full_fn)['scores']
            s = np.clip(s, -1, 1)
            scores.append(s)

        fig, ax = plt.subplots(nrows=len(scores))
        for i in range(len(scores)):
            ax[i].hist(scores[i], range=(-1, 1), bins=20)
            ax[i].set_title(f'histogram {i}')
        fig.suptitle(obj)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dataset_dir = 'data/gag/grasps/'
    # create_two_finger_grasp_annotations(dataset_dir)
    plot_score_distribution(dataset_dir, 2)

