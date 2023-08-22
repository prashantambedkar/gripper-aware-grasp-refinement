import os
import numpy as np

import burg_toolkit as burg

from convonets.src.data.core import Field
from convonets.src.utils import binvox_rw
from convonets.src.common import coord2index, normalize_coord
from gag_refine.dataset import pre_normalise_points, load_object_library, balanced_sample_sizes
from gag_refine.utils.transform import transform_points


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        return True

# 3D Fields
class PatchPointsField(Field):
    ''' Patch Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape and then split to patches.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        
    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        # if self.multi_files is None:
        if True:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        # acquire the crop
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['query_vol'][0][i])
                     & (points[:, i] <= vol['query_vol'][1][i]))
        ind = ind_list[0] & ind_list[1] & ind_list[2]
        data = {None: points[ind],
                    'occ': occupancies[ind],
            }
            
        if self.transform is not None:
            data = self.transform(data)

        # calculate normalized coordinate w.r.t. defined query volume
        p_n = {}
        for key in vol['plane_type']:
            # projected coordinates normalized to the range of [0, 1]
            p_n[key] = normalize_coord(data[None].copy(), vol['input_vol'], plane=key)
        data['normalized'] = p_n

        return data

class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        # added for GAG scenes
        points = pre_normalise_points(points)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class PointsSDFField(Field):
    """ Point Field with SDF values as well.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    """
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None,
                 clamp_sdf=None, clamp_margin=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files
        self.clamp_sdf = clamp_sdf
        self.clamp_margin = clamp_margin or 0

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-6 * np.random.randn(*points.shape)

        # added for GAG scenes
        points = pre_normalise_points(points)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        sdf_values = points_dict['sdf']
        if sdf_values.dtype == np.float16:  # also break symmetry... although not exactly sure what it means
            sdf_values = sdf_values.astype(np.float32)
            sdf_values += 1e-6 * np.random.randn(*sdf_values.shape)

        # clamp the values to a max distance, then scale to [-1, 1] (we have to check if this is a good idea)
        if self.clamp_sdf is not None:
            sdf_values = np.clip(sdf_values, -self.clamp_sdf, self.clamp_sdf)
            if self.clamp_margin is not None:
                norm = self.clamp_sdf * (1 + self.clamp_margin)
                sdf_values /= norm

        data = {
            None: points,
            'occ': occupancies,
            'sdf': sdf_values,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


class VoxelsField(Field):
    ''' Voxel field class.

    It provides the class used for voxel-based data.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        file_path = os.path.join(model_path, self.file_name)

        with open(file_path, 'rb') as f:
            voxels = binvox_rw.read_as_3d_array(f)
        voxels = voxels.data.astype(np.float32)

        if self.transform is not None:
            voxels = self.transform(voxels)

        return voxels

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PatchPointCloudField(Field):
    ''' Patch point cloud field.

    It provides the field used for patched point cloud data. These are the points
    randomly sampled on the mesh and then partitioned.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, transform_add_noise=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, vol):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            vol (dict): precomputed volume info
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        # add noise globally
        if self.transform is not None:
            data = {None: points, 
                    'normals': normals}
            data = self.transform(data)
            points = data[None]

        # acquire the crop index
        ind_list = []
        for i in range(3):
            ind_list.append((points[:, i] >= vol['input_vol'][0][i])
                    & (points[:, i] <= vol['input_vol'][1][i]))
        mask = ind_list[0] & ind_list[1] & ind_list[2]# points inside the input volume
        mask = ~mask # True means outside the boundary!!
        data['mask'] = mask
        points[mask] = 0.0
        
        # calculate index of each point w.r.t. defined resolution
        index = {}
        
        for key in vol['plane_type']:
            index[key] = coord2index(points.copy(), vol['input_vol'], reso=vol['reso'], plane=key)
            if key == 'grid':
                index[key][:, mask] = vol['reso']**3
            else:
                index[key][:, mask] = vol['reso']**2
        data['ind'] = index
        
        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete

class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        # added for GAG scenes
        points = pre_normalise_points(points)
        
        data = {
            None: points,
            'normals': normals,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PartialPointCloudField(Field):
    ''' Partial Point cloud field.

    It provides the field used for partial point cloud data. These are the points
    randomly sampled on the mesh and a bounding box with random size is applied.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
        part_ratio (float): max ratio for the remaining part
    '''
    def __init__(self, file_name, transform=None, multi_files=None, part_ratio=0.7):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.part_ratio = part_ratio

    def load(self, model_path, idx, category):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        
        side = np.random.randint(3)
        xb = [points[:, side].min(), points[:, side].max()]
        length = np.random.uniform(self.part_ratio*(xb[1] - xb[0]), (xb[1] - xb[0]))
        ind = (points[:, side]-xb[0])<= length
        data = {
            None: points[ind],
            'normals': normals[ind],
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.
        
        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class GraspsField(Field):
    def __init__(self, dataset_dir, n_fingers=(2, 3, 4, 5), n_sample=None, clamp_fc=None, clamp_margin=None,
                 contact_noise=None):
        """ Grasp annotations

        Args:
            dataset_dir (str):
            n_fingers (Iterable): number of fingers, will choose randomly from this Iterable each time sth is loaded
            n_sample (int): number of samples to gather when loading, if all use None
        """
        self.object_library = load_object_library(dataset_dir)
        self.n_fingers = n_fingers
        self.n_sample = n_sample
        self.clamp_fc = clamp_fc
        self.clamp_margin = clamp_margin
        self.contact_noise = contact_noise

    def load(self, model_path, idx, category):
        """ Loads the data point.

        Args:
            model_path (str): path to model  (i.e. scene directory)
            idx (int): ID of data point
            category (int): index of category
        """
        # load the scene from file
        yaml_fn = os.path.join(model_path, 'scene.yaml')
        scene, _, _ = burg.Scene.from_yaml(yaml_fn, object_library=self.object_library)

        # find grasp directory based on model_path
        grasp_dir = os.path.join(model_path, '../../grasps/')
        n_fingers = np.random.choice(self.n_fingers)  # randomly choose each time the actual number of fingers

        # prepare lists to gather grasps
        contact_points_list = []
        scores_list = []

        # prepare subsampling if desired
        if self.n_sample is None:
            n_samples_per_obj = np.full(len(scene.objects), fill_value='all')
        else:
            n_samples_per_obj = balanced_sample_sizes(self.n_sample, len(scene.objects))

        # loop over objects in scene
        for n_samples_this_obj, obj_instance in zip(n_samples_per_obj, scene.objects):
            identifier = obj_instance.object_type.identifier
            obj_grasp_dir = os.path.join(grasp_dir, identifier, f'{n_fingers}')
            pose = obj_instance.pose

            # we have multiple grasp files per obj in which different types of grasps (qualities) are stored
            # they should be balanced, to have positive, hard negative and negative grasps
            files = os.listdir(obj_grasp_dir)
            if n_samples_this_obj == 'all':
                n_samples_per_file = np.full(len(files), fill_value='all')
            else:
                n_samples_per_file = balanced_sample_sizes(n_samples_this_obj, len(files))

            # gather grasps from each file
            for n_samples_this_file, file in zip(n_samples_per_file, files):
                f = os.path.join(obj_grasp_dir, file)
                data = np.load(f)
                contact_points = data['contact_points']
                scores = data['scores']
                if n_samples_this_file != 'all':
                    replace = n_samples_this_file > len(contact_points)  # catch over-sampling case
                    idcs = np.random.choice(len(contact_points), n_samples_this_file, replace=replace)
                    contact_points = contact_points[idcs]
                    scores = scores[idcs]

                sigma = self.contact_noise
                if contact_points.dtype == np.float16:
                    contact_points = contact_points.astype(np.float32)
                    if sigma is None:
                        sigma = 1e-6
                contact_points += sigma * np.random.randn(*contact_points.shape)

                # todo: augmentation. add some noise specifically along contact normal?
                contact_points = transform_points(pose, contact_points)  # object's pose in the scene
                contact_points = pre_normalise_points(contact_points)  # pre-normalisation for convonet
                contact_points_list.append(contact_points)

                # clamp the scores, then possibly scale to [-1, 1]
                if self.clamp_fc is not None:
                    scores = np.clip(scores, -self.clamp_fc, self.clamp_fc)
                    if self.clamp_margin is not None:
                        norm = self.clamp_fc * (1 + self.clamp_margin)
                        scores /= norm
                scores_list.append(scores)

        data = {
            'contact_points': np.concatenate(contact_points_list, axis=0),
            'scores': np.concatenate(scores_list, axis=0),
        }
        return data


class MetaDataField(Field):
    ''' Provides information about the scene.'''

    def load(self, model_path, idx, category):
        ''' Loads the meta data.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        scene_idx = os.path.basename(model_path)
        meta = {
            'scene_idx': scene_idx,
            'scene_dir': model_path,
            'scene_fn': os.path.join(model_path, 'scene.yaml'),
            'dataset_dir': os.path.join(model_path, '../../'),
        }
        return meta

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True