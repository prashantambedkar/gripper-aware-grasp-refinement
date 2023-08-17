import copy
import os

import numpy as np
import burg_toolkit as burg
burg.visualization.configure_visualizer_mode(burg.visualization.VisualizerMode.CONFIGURABLE_VIEWER)

from gag_refine.dataset import manifold
from gag_refine.dataset import creation as dataset_creation


synsets_to_use = {
    '02691156': 'airplane,aeroplane,plane',
    '02876657': 'bottle',
    '02880940': 'bowl',
    '02924116': 'bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi',
    '02942699': 'camera,photographic camera',
    '02946921': 'can,tin,tin can',
    '02954340': 'cap',
    '02958343': 'car,auto,automobile,machine,motorcar',
    '02992529': 'cellular telephone,cellular phone,cellphone,cell,mobile phone',
    '03001627': 'chair',
    '03046257': 'clock',
    '03085013': 'computer keyboard,keypad',
    '03211117': 'display,video display',
    '03261776': 'earphone,earpiece,headphone,phone',
    '03325088': 'faucet,spigot',
    '03513137': 'helmet',
    '03593526': 'jar',
    '03624134': 'knife',
    '03636649': 'lamp',
    '03642806': 'laptop,laptop computer',
    '03691459': 'loudspeaker,speaker,speaker unit,loudspeaker system,speaker system',
    '03759954': 'microphone,mike',
    '03797390': 'mug',
    '03991062': 'pot,flowerpot',
    '04004475': 'printer,printing machine',
    '04074963': 'remote control,remote',
    '04225987': 'skateboard',
    '04401088': 'telephone,phone,telephone set',
}


def sample_object_library(dataset_dir, shapenet_dir, n_shapes_per_synset=40):
    lib_dir = os.path.join(dataset_dir, 'object_library')
    burg.io.make_sure_directory_exists(lib_dir)
    lib = burg.ObjectLibrary(name='ShapeNetCore V2 Subset',
                             description='only visual objects, not intended for grasping')
    lib.to_yaml(yaml_fn=os.path.join(lib_dir, 'object_library.yaml'))

    for synset, synset_name in synsets_to_use.items():
        # find available shapes in directory
        synset_dir = os.path.join(shapenet_dir, synset)
        shapes = os.listdir(synset_dir)
        print(f'{len(shapes)} shapes found in synset {synset} - {synset_name}')
        assert len(shapes) >= n_shapes_per_synset, 'synset does not have enough shapes'

        # select shapes
        selected_shapes = np.random.choice(shapes, n_shapes_per_synset, replace=False)

        # some shapes are in multiple synsets. if we get one that we have already, just resample
        while np.any([str(shape) in lib.keys() for shape in selected_shapes]):
            print('shape is already in library, resampling')
            selected_shapes = np.random.choice(shapes, n_shapes_per_synset, replace=False)

        # put them in object library
        for shape in selected_shapes:
            shape = str(shape)
            mesh_fn = os.path.join(synset_dir, shape, 'models/model_normalized.obj')
            assert os.path.isfile(mesh_fn), f'mesh file assumed incorrectly: {mesh_fn}'
            lib_entry = burg.ObjectType(shape, mesh_fn=mesh_fn, name=synset_name)
            # burg.visualization.show_geometries([lib_entry.mesh])
            assert shape not in lib.keys(), f'shape {shape} is already in the library. {lib[shape]}'
            lib[shape] = lib_entry

        lib.to_yaml()

    return lib


def create_scenes(scenes_dir, lib, scene_size=0.297):
    """
        object will be randomly rotated along all axes!
        axis-aligned bounding box will be computed (aabb)
        scale object so that longest side length of aabb is between 0.5 and 0.8 x 0.297m
        update object library!!
        translate object so that center of aabb is at 0.297/2 for all axes
    """
    new_object_library = burg.ObjectLibrary(lib.name, lib.description)  # to save modified shapes [scale]

    for k, (identifier, shape) in enumerate(lib.items()):
        print(f'{k}/{len(lib)}: {identifier} ({shape.name})')

        # we load the mesh manually first, as we will change the scale of the object type
        # once the object type loads its mesh, it does not update
        mesh = burg.io.load_mesh(mesh_fn=shape.mesh_fn)

        # apply random rotation
        random_pose = burg.sampling.random_poses(1).reshape(4, 4)
        random_pose[:3, 3] = 0  # no translation, just rotation
        mesh.transform(random_pose)

        # figure out scale.
        # max extent should be in factor_range of scene_size
        factor = np.random.uniform(0.7, 1.0)
        max_extent = mesh.get_axis_aligned_bounding_box().get_max_extent()
        scale = factor*scene_size/max_extent

        # update object library
        # scale is not settable, so we actually need to create a new ObjectType
        new_shape = burg.ObjectType(identifier=identifier, name=shape.name, mesh_fn=shape.mesh_fn, scale=scale)
        new_object_library[identifier] = new_shape

        # now we can load object mesh with correct scale and planned orientation
        instance = burg.ObjectInstance(new_shape, random_pose)

        # place at random position such that it fits into the scene
        aabb = instance.get_mesh().get_axis_aligned_bounding_box()
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()

        lower_rng_bound = -min_bound
        upper_rng_bound = scene_size-max_bound
        for i in range(len(lower_rng_bound)):
            if lower_rng_bound[i] > upper_rng_bound[i]:
                lower_rng_bound[i], upper_rng_bound[i] = upper_rng_bound[i], lower_rng_bound[i]

        position = np.random.uniform(lower_rng_bound, upper_rng_bound)

        # # check lower and upper rng bounds visually
        # low_pose = random_pose
        # low_pose[:3, 3] = lower_rng_bound
        # instance1 = burg.ObjectInstance(new_shape, low_pose)
        # mesh1 = instance1.get_mesh()
        # print('center1:', mesh1.get_center())
        #
        # high_pose = copy.deepcopy(random_pose)
        # high_pose[:3, 3] = upper_rng_bound
        # instance2 = burg.ObjectInstance(new_shape, high_pose)
        # mesh2 = instance2.get_mesh()
        # print('center2:', mesh2.get_center())

        final_pose = copy.deepcopy(random_pose)
        final_pose[:3, 3] = position
        instance = burg.ObjectInstance(new_shape, final_pose)

        scene = burg.Scene(ground_area=(scene_size, scene_size), objects=[instance])
        # burg.visualization.show_geometries([scene])

        scene_dir = os.path.join(scenes_dir, identifier)
        burg.io.make_sure_directory_exists(scene_dir)
        scene.to_yaml(os.path.join(scene_dir, 'scene.yaml'), object_library=lib)

    # finally we can save new object library. this overwrites the old library, but as there are only changes in scale
    # and the scale is not used in this function, that should be fine.
    print('writing adjusted object library to file...')
    new_object_library.to_yaml(lib.filename)
    print('done.')


def create_splits(scenes_dir, lib, train=36, val=2, test=2):
    print('creating splits...')
    assert train+val+test == 40

    # create dict with all shapes that correspond to each category
    print('scanning object library...')
    all_shapes_per_category = {}
    for identifier, shape in lib.items():
        category = shape.name
        if category not in all_shapes_per_category.keys():
            all_shapes_per_category[category] = []
        all_shapes_per_category[category].append(identifier)
    print('all_shapes_per_category listed')

    # it should be 40 for all 28 shapes
    assert len(all_shapes_per_category) == 28
    for cat, shapes in all_shapes_per_category.items():
        assert len(shapes) == 40
    print('assertions complete.')

    # write into files
    # shapes were already shuffled during selection, so no shuffling required here
    split_names = ['train', 'val', 'test']
    numbers = [train, val, test]
    for split_name, number in zip(split_names, numbers):
        split_fn = os.path.join(scenes_dir, f'{split_name}.lst')
        print(f'starting file write for split {split_name}: {number}...')
        with open(split_fn, 'w') as split_file:
            for _, shapes in all_shapes_per_category.items():
                for _ in range(number):
                    split_file.write(f'{shapes.pop()}\n')
        print(f'file write complete for {split_name} split.')


def debug_occupancy(scenes_dir, lib, shape_name):
    from gag_refine.dataset.occupancy import sample_points, get_occupancy_map

    scene_padding = 0.012  # sample points with some padding, set padding in ConvONet to 0.08
    object_padding = 0.03  # some more padding around the objects, but we limit it to the scene padding

    scene_dir = os.path.join(scenes_dir, shape_name)
    scene, _, _ = burg.Scene.from_yaml(os.path.join(scene_dir, 'scene.yaml'), lib)

    # get bounding box and sample points within
    mesh = scene.objects[0].get_mesh()
    trimesh = burg.mesh_processing.as_trimesh(mesh)
    print(f'watertight? {trimesh.is_watertight}')
    lower_bounds = mesh.get_min_bound() - object_padding
    upper_bounds = mesh.get_max_bound() + object_padding
    points = sample_points(30000, lower_bounds, upper_bounds)
    points = points[points[:, 2] > -scene_padding]  # filter out those that are too low in z

    print(f'checking occupancy for {len(points)} points')
    occupancy_map = get_occupancy_map(points, scene, check_z=False)
    hist = np.unique(occupancy_map, return_counts=True)
    print(hist)

    occ_labels = (occupancy_map >= 0).astype(bool)
    occ_points = [points[occ_labels], points[~occ_labels]]

    burg.visualization.show_geometries([scene, *occ_points])


def browse_shapes(name, object_library):
    # show all shapes that have the keyword in the name (shapenet category)
    shapes = [shape for key, shape in object_library.items() if name in shape.name]
    print(f'browsing {len(shapes)} shapes with keyword "{name}"')
    for i, shape in enumerate(shapes):
        print(f'{i}/{len(shapes)}: {shape.identifier}\t\t{shape.name}')
        burg.visualization.show_geometries([shape.mesh])


def main():
    dataset_dir = 'c:/users/rudorfem/datasets/shapenet_core_v2/'
    shapenet_dir = 'c:/users/rudorfem/datasets/ShapeNetCore.v2/ShapeNetCore.v2/'

    scenes_dir = os.path.join(dataset_dir, 'scenes')
    burg.io.make_sure_directory_exists(scenes_dir)

    # USE THIS FOR CREATING NEW OBJECT LIBRARY - WILL CREATE NEW SUBSET OF SHAPENET SHAPES
    # lib = sample_object_library(dataset_dir, shapenet_dir)
    # create_splits(scenes_dir, lib, 36, 2, 2)

    # START FROM HERE IF LIBRARY IS SET
    lib = burg.ObjectLibrary.from_yaml(os.path.join(dataset_dir, 'object_library/object_library.yaml'))
    print(f'got object library with {len(lib)} items')

    # creates meshes and updates
    # manifold.make_meshes_watertight(lib)

    # THIS RECREATES NEW (RANDOM) SCENES
    # create_scenes(scenes_dir, lib)
    # # todo: reload object library after this. scale of object types got adjusted.
    # lib = burg.ObjectLibrary.from_yaml(os.path.join(dataset_dir, 'object_library/object_library.yaml'))
    # print(f'reloaded object library with {len(lib)} items')

    # THIS RECREATES POINT CLOUDS AND ANNOTATIONS (some randomness, but should not affect training much)
    # dataset_creation.create_full_point_cloud_data(
    #     scenes_dir, lib, splits=['train', 'val', 'test'],
    #     ground_factor=0, with_ply=True, with_semantics=False, n_points=100000)
    # dataset_creation.create_annotated_query_points(
    #     scenes_dir, lib, splits=['train', 'val', 'test'], with_sdf=True,
    #     n_points_whole_scene=50000, n_points_per_object_bb=30000, ground_plane=False,
    #     with_semantics=False, n_from_point_cloud=50000, from_point_cloud_noise=0.02)

    # debug_occupancy(scenes_dir, lib, '273e1da4dfcf82443c5e2d7fd3020266')
    # dataset_creation.inspect_scene_annotations(scenes_dir, lib, '273e1da4dfcf82443c5e2d7fd3020266')
    # n_show_objs = 20
    # obj_ids = np.random.choice([key for key in lib.keys()], n_show_objs, replace=False)
    # for i in range(n_show_objs):
    #     #     debug_occupancy(scenes_dir, lib, obj_ids[i])
    #     dataset_creation.inspect_scene_annotations(scenes_dir, lib, obj_ids[i])
    # browse_shapes('micro', lib)
    dataset_creation.print_occupancy_summary(scenes_dir, splits=['train', 'val', 'test'])


if __name__ == '__main__':
    main()
