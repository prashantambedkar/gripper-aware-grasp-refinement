from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from scipy.spatial import ConvexHull
import burg_toolkit as burg


def compute_stats(ground_truth, predictions, zeros_instead_of_nans=False):
    """ Computes CCR, Precision and Recall based on Ferrari-Canny grasp scores

    Args:
        ground_truth (tensor): ferrari canny GT scores
        predictions (tensor): ferrari canny predicted scores
        zeros_instead_of_nans (bool): if True, will return 0 instead of nan in case of zero division

    Returns: dict
    """
    tp = ((ground_truth > 0) & (predictions > 0)).sum().item()
    fp = ((ground_truth <= 0) & (predictions > 0)).sum().item()
    fn = ((ground_truth > 0) & (predictions <= 0)).sum().item()
    tn = ((ground_truth <= 0) & (predictions <= 0)).sum().item()
    n_total = torch.numel(ground_truth)
    ccr = (tp + tn) / n_total

    no_zero_division_error = True
    # precision
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0 if zeros_instead_of_nans else np.nan
        no_zero_division_error = False

    # recall
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0 if zeros_instead_of_nans else np.nan
        no_zero_division_error = False

    # f1 score
    if no_zero_division_error and tp > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0 if zeros_instead_of_nans else np.nan

    results = {
        'ccr': ccr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    return results


class FerrariCanny:
    def __init__(self, mu=0.5, n_edges=8, soft_contact=True, gamma=0.01, soft_contact_method='full',
                 return_negative_scores=True):
        self.mu = mu
        self.n_edges = n_edges
        self.soft_contact = soft_contact
        self.gamma = gamma
        self.debug = False
        self.hull_time = 0.0
        self.soft_contact_method = soft_contact_method
        self.return_negative_scores = return_negative_scores

    def get_friction_cone_points(self, normals):
        """
        gets the points on the friction cone with f_n = 1, considering mu and n_edges, transforms it into the correct
        orientation according to the inward-facing normal(s)

        params:
        normals - (3,) or (N, 3) array of inward-facing normals
        """
        normals = np.reshape(normals, (-1, 3))
        # exploiting look_at function to get transformation matrices, which we can then apply to the template cone pts
        # looking from origin in direction of the surface normals (z-axis)
        transforms = burg.util.look_at([0, 0, 0], normals)
        transforms = np.reshape(transforms, (-1, 4, 4))  # ensure unsqueezed form

        # build friction cone points using x/y axes as tangent vectors
        cone_points = np.zeros((len(normals), self.n_edges, 3))
        for i in range(self.n_edges):
            tan_vecs = transforms[:, :3, 0] * np.cos(2 * np.pi * i / self.n_edges) + \
                      transforms[:, :3, 1] * np.sin(2 * np.pi * i / self.n_edges)
            cone_points[:, i, :] = transforms[:, :3, 2] + self.mu * tan_vecs

        if self.debug:
            print(f'shape of transforms: {transforms.shape}')
            print(f'first normal: {normals[0]}')
            print(f'z-axis of first tf: {transforms[0, :3, 2]}')
            print('(they should be the same)')
            print(f'shape of cone points: {cone_points.shape}')  # n_normals x n_edges x 3
            burg.visualization.show_geometries([np.reshape(cone_points, (-1, 3)), np.eye(3)])

        return cone_points

    def compute_L1_score(self, contacts, normals, com):
        """
        computes the Ferrari Canny L1 score of the grasp

        params:
        contacts - Nx3 ndarray with contact positions
        normals - Nx3 ndarray with inward facing, normalised surface normals
        com - 3, ndarray position of center of mass of object
        """
        contacts, normals, com = np.asarray(contacts), np.asarray(normals), np.asarray(com)
        assert contacts.shape[0] == normals.shape[0]
        assert contacts.shape[1] == normals.shape[1] == com.shape[0] == 3
        n_contacts = contacts.shape[0]

        # approximate the friction cone by a set of forces on its hull
        cone_forces = self.get_friction_cone_points(normals)  # shape: n_contacts, self.n_edges, 3

        # compute the corresponding torques
        # todo: check if we need some torque-scaling as in dex-net?
        wrenches = np.zeros((n_contacts, self.n_edges, 6))
        wrenches[:, :, 0:3] = cone_forces
        moment_arms = contacts - com  # n_contacts, 3
        wrenches[:, :, 3:] = np.cross(moment_arms[:, None, :], cone_forces, axis=-1)

        if self.soft_contact:
            """
            The soft finger contact model introduces torsional friction around the normal. The moment can be estimated
            by computing gamma * normal_force, where gamma in theory depends on the area and friction coefficient. 
            With a normal_force of 1, the moment's intensity is +-gamma, as the friction works in both directions.
            The wrenches we have so far do exert a normal_force, but lack the corresponding torsional moment.
            Wrenches can be superposed as long as they are written in the same coordinate frame. By multiplication with
            the normal, we can transform the torsional moment into the object frame and add it to the existing wrenches.
            We would need to add it to each wrench once with positive and once with negative sign, doubling the amount 
            of wrenches. This is method 'full', the most accurate one. But it can lead to high computation time for the 
            convex hull due to the number of wrenches. An approximation with fewer number of wrenches can therefore be
            desired.
            In dex-net, they introduce two new elementary wrenches with just the moment. This is method 'elementary'. 
            The moment however can only be exerted in the presence of a corresponding normal force, therefore the
            elementary wrench is actually outside of the wrench cone. Carpin et al. (2016) state that using elementary 
            wrenches sufficiently approximates the convex hull and refer to Murray (1994), but I can't find any such 
            statement in the referenced book. As I see it, using elementary wrenches might both over- or underestimate
            the actual convex hull.
            As an alternative approximation, without introducing any new wrenches (thus saving more computation time), 
            we add the torsional moment to all the existing wrenches, but with alternating sign (method: 'alternating').
            Because the alternating wrenches are a proper subset of the full wrenches, this method always computes a 
            conservative, smaller approximation of the convex hull. With high number of n_edges, this could be the
            preferred method, but for small number of edges it might be unreliable.
            """
            if self.soft_contact_method == 'alternating':
                wrenches[:, 0::2, 3:] += (self.gamma * normals)[:, None, :]
                wrenches[:, 1::2, 3:] -= (self.gamma * normals)[:, None, :]
            elif self.soft_contact_method == 'full':
                wrenches = np.repeat(wrenches, 2, axis=0)
                wrenches[0::2, :, 3:] += (self.gamma * normals)[:, None, :]
                wrenches[1::2, :, 3:] -= (self.gamma * normals)[:, None, :]
            elif self.soft_contact_method == 'elementary':
                # for each finger, add two elementary wrenches with just the moment
                nw = wrenches.shape[0]
                elementary_wrenches = np.zeros((nw, 2, 6))
                elementary_wrenches[:, 0, 3:] = self.gamma * normals
                elementary_wrenches[:, 1, 3:] = -self.gamma * normals
                wrenches = np.concatenate([wrenches, elementary_wrenches], axis=1)
            else:
                raise NotImplementedError(f'soft_contact_method can be [full | alternating | elementary], '
                                          f'but got {self.soft_contact_method}')

        # compute the convex hull. scipy internally uses qhull
        try:
            self.hull_time -= timer()  # subtract starting time, as we later add end time
            wrenches = wrenches.reshape(-1, 6)
            hull = ConvexHull(wrenches.reshape(-1, 6))
        except:
            # print('ERROR convex hull could not be computed.')
            return -np.inf if self.return_negative_scores else 0
        finally:
            self.hull_time += timer()
        if len(hull.vertices) == 0:
            # print('ERROR convex hull does not have any vertices.')
            return -np.inf if self.return_negative_scores else 0

        # conveniently, the hull also contains equations for every hyperplane on which the facets lie
        # the equations consist of normal vector eq[:-1] and offset eq[-1]
        # see https://stackoverflow.com/a/42165596/1264582
        # a point is inside the hull, if normal dot point + offset < 0
        # since we consider origin, we can directly check for the offset and this will also give us the distance
        # if all offsets are < 0, origin is enclosed, and the smallest absolute offset is the grasp quality
        quality = -np.max([eq[-1] for eq in hull.equations])
        if self.debug:
            print(f'score is: {quality:.05f} (force closure if > 0)')

        return quality if self.return_negative_scores else np.max(quality, 0)

    @classmethod
    def _parallel_computation_helper(cls, args):
        contacts, normals, com, kwargs = args
        fc = cls(**kwargs)
        return fc.compute_L1_score(contacts, normals, com)

    @classmethod
    def compute_L1_scores_parallel(cls, contacts, normals, com, **kwargs):
        """
        computes the Ferrari Canny L1 score of the given grasps
        there are M grasps with N contact points each - no possibility to combine different number of contact points

        params:
        contacts - MxNx3 ndarray with contact positions
        normals - MxNx3 ndarray with inward facing, normalised surface normals
        com - Mx3 or 3, ndarray position of center of mass of object
        kwargs - additional parameters that can be passed to configure FerrariCanny class
        """
        # make sure input is nice
        assert contacts.shape[0] == normals.shape[0], 'must be same number of grasps'
        assert contacts.shape[1] == normals.shape[1], 'must be same number of contact points'
        n_grasps = contacts.shape[0]
        if com.ndim == 1:
            com = np.repeat(com[None, :], n_grasps, axis=0)
        elif com.ndim == 2:
            assert com.shape[0] == n_grasps
        else:
            raise ValueError(f'unexpected com shape {com.shape}')

        # build a generator for input, since we can only pass one value to parallel function
        args = ((contacts[i], normals[i], com[i], kwargs) for i in range(n_grasps))
        pool = ThreadPoolExecutor()
        grasp_scores = list(pool.map(cls._parallel_computation_helper, args))
        return np.array(grasp_scores)


def show_me_some_friction_cones():
    fc = FerrariCanny(mu=0.5, n_edges=8)
    normals = np.eye(3)
    print(f'friction cone points for {normals[0]}')
    fc.get_friction_cone_points(normals[0])
    print(f'friction cone points for {normals}')
    fc.get_friction_cone_points(normals)

    fc = FerrariCanny(mu=0.2, n_edges=12)
    normals = np.eye(3)
    print(f'friction cone points for {normals[0]}')
    fc.get_friction_cone_points(normals[0])
    print(f'friction cone points for {normals}')
    fc.get_friction_cone_points(normals)


def simple_examples():
    fc = FerrariCanny(soft_contact=False)
    contacts = np.array([[0, 0, 1], [0, 0.5, -1], [0, -0.5, -1]])
    normals = np.array([[0, 0, -1], [0, 0, 1], [0, 0, 1]])
    fc.compute_L1_score(contacts, normals, [0, 0, 0])

    fc = FerrariCanny(soft_contact=False)
    contacts = np.array([[0.0, 0, 1], [0, 0.0, -1]])
    normals = np.array([[0, 0, -1], [0, 0, 1]])
    fc.compute_L1_score(contacts, normals, [0.5, 0.2, 0.1])

    fc = FerrariCanny(soft_contact=True)
    contacts = np.array([[0.0, 0, 1], [0, 0.0, -1]])
    normals = np.array([[0, 0, -1], [0, 0, 1]])
    fc.compute_L1_score(contacts, normals, [0.5, 0.2, 0.1])


def fc_helper(args):
    contacts, normals, com = args
    fc = FerrariCanny()
    fc.debug = False
    return fc.compute_L1_score(contacts, normals, com)


def check_computation_speed():
    fc = FerrariCanny(soft_contact=True)
    fc.debug = False
    n_grasps = 10000
    n_fingers = 3
    contacts = np.random.random((n_grasps, n_fingers, 3))
    normals = np.random.random((n_grasps, n_fingers, 3))
    normals = normals / np.linalg.norm(normals, axis=-1)[:, :, None]

    print(f'computing {n_grasps} grasps with {n_fingers} contacts sequentially with same FC object instance')
    origin = np.zeros(3)
    # scores = np.zeros(n_grasps)
    # start_time = timer()
    # for i in range(len(contacts)):
    #     scores[i] = fc.compute_L1_score(contacts[i], normals[i], origin)
    # end_time = timer()
    # print(f'computed scores in {end_time-start_time:.2f} sec, i.e. {n_grasps/(end_time-start_time):.2f} grasps/sec')
    # print(f'of which {fc.hull_time:.2f} secs were used for computing the convex hull')
    # print(f'force-closure grasps: {np.count_nonzero(scores)}')
    # print(f'*'*10)

    print(f'computing {n_grasps} grasps with {n_fingers} contacts in parallel with ThreadPoolExecutor')
    args = ((contacts[i], normals[i], origin) for i in range(n_grasps))
    pool = ThreadPoolExecutor()
    start_time = timer()
    grasp_scores = list(pool.map(fc_helper, args))
    end_time = timer()
    scores = np.array(grasp_scores)
    print(
        f'computed scores in {end_time - start_time:.2f} sec, i.e. {n_grasps / (end_time - start_time):.2f} grasps/sec')
    print(f'force-closure grasps: {np.count_nonzero(scores)}')
    print(f'*' * 10)

    print(f'computing {n_grasps} grasps with {n_fingers} contacts in with FerrariCanny class thingy')
    start_time = timer()
    scores = FerrariCanny.compute_L1_scores_parallel(contacts, normals, origin)
    end_time = timer()
    print(
        f'computed scores in {end_time - start_time:.2f} sec, i.e. {n_grasps / (end_time - start_time):.2f} grasps/sec')
    print(f'force-closure grasps: {np.count_nonzero(scores)}')
    print(f'*' * 10)


def main():
    # we can try ferrari canny L1 or Linf, the latter uses Minkowski sum and is more computationally heavy, but maybe
    # more meaningful especially considering that we have grasps with varying number of fingers?
    # let's try the other one first, then see what the runtime bottleneck is
    # https://www.wikiwand.com/en/Minkowski_addition

    # note that the L1 with c1, c2 is at most as good as L1 with c1, c2, and some other contact point c3
    # since we compute the convex hull, it cannot get worse; the system should therefore automatically give
    # grasps with higher number of fingers a better score

    # what exactly is the range of values we will encounter? max 1, but probably much smaller in reality

    # input:
    # get contact points c1...cm
    # get surface normals n1...nm
    # get center of mass z

    # setup approximated friction cones
    # i.e. assume normal force = 1; assume mu = 0.5; assume num of edges = 8-10
    # construct a wrench vector for each force_edge f [f, (c-z) x f] - f is the point on the friction cone disc
    # - compute forces (= points of the friction cone)
    # - compute torques (= moment arm x points of the friction cone)
    #   ==> these two build one column in the grasp matrix
    # we chose an arbitrary normal force=1, but we need to choose an arbitrary normal moment as well... which depends
    # on the scale, i.e. the length of the moment arm
    # what happens if we make this too small??
    #   - moment arms might be tiny compared to the forces, i.e. torques are tiny and will limit the convex hull
    #   - if too big, the moments will not have any effect on the convex hull...
    # but should it depend on the size of the object?? no i don't think so.
    # let's compute the CH without any scaling factor, then we can visualise both the moment's hull and the force's
    # hull and compare scales with e.g. a unit cube

    # - compute pure moment (= ?)
    # optional: if SOFT CONTACT, add wrenches for moment as well! assume gamma, assume some fingertip radius
    # ==> it can only increase the convex hull, so no harm done by wrong parameter settings

    # build grasp matrix, i.e. union of all wrenches  # alternatively: minkowski sum for Linf instead of L1
    # compute convex hull
    # check origin is within convex hull --> force closure; else not
    # check if origin is on the boundary (?)
    # find minimum norm vector across all facets of convex hull... how???
    # this is the ferrari canny score!

    # implementations to refer to:
    # https://gitlab.com/Simox/simox/-/blob/master/GraspPlanning/GraspQuality/GraspQualityMeasureWrenchSpace.cpp
    #   they don't seem to consider the torques at all???
    # dex-net
    # show_me_some_friction_cones()
    check_computation_speed()
    # simple_examples()
    pass


if __name__ == '__main__':
    main()
