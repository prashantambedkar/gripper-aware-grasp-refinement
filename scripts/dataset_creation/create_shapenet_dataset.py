import os

import numpy as np
import burg_toolkit as burg

# from gag_refine.dataset.creation import create_full_point_cloud_data, create_annotated_query_points


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


def main():
    dataset_dir = 'c:/users/rudorfem/datasets/shapenet_core_v2/'
    shapenet_dir = 'c:/users/rudorfem/datasets/ShapeNetCore.v2/ShapeNetCore.v2/'

    # lib = sample_object_library(dataset_dir, shapenet_dir)
    lib = burg.ObjectLibrary.from_yaml(os.path.join(dataset_dir, 'object_library/object_library.yaml'))
    print(f'got object library with {len(lib)} items')

    # create_full_point_cloud_data(base_dir, lib)
    # create_annotated_query_points(base_dir, lib, with_sdf=True, n_points_whole_scene=100000,
    #                               n_points_per_object_bb=30000, ground_plane=True)


if __name__ == '__main__':
    main()
