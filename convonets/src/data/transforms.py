import logging

import numpy as np


logger = logging.getLogger(__name__)


# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        indices = np.random.randint(points.shape[0], size=self.N)
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']
        data_out = data.copy()

        if isinstance(self.N, int):
            # simply sampling N items
            idx = np.random.choice(points.shape[0], size=self.N, replace=False)
            data_out.update({k: v[idx] for k, v in data_out.items()})
        else:
            # balancing classes inside/outside of objects
            Nt_out, Nt_in = self.N
            occ_binary = (occ >= 0.5)
            in_mask = occ_binary
            out_mask = ~occ_binary

            replace_out = np.count_nonzero(out_mask) < Nt_out
            replace_in = np.count_nonzero(in_mask) < Nt_in
            if replace_in or replace_out:
                logger.warning(f'oversampling minority class.')

            idx_out = np.random.choice(np.count_nonzero(out_mask), size=Nt_out, replace=replace_out)
            idx_in = np.random.choice(np.count_nonzero(in_mask), size=Nt_in, replace=replace_in)

            data_out.update(
                {k: np.concatenate([v[in_mask][idx_in], v[out_mask][idx_out]], axis=0) for k, v in data_out.items()}
            )
            # not sure what for... convonet legacy
            data_out.update({'volume': (occ_binary.sum() / len(occ_binary)).astype(np.float32)})

        return data_out
