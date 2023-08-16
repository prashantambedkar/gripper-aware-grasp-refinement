import torch


class GramSchmidtRotationMapping:
    """
    Maps rotation matrices to a 6D representation that is continuous and hence more suitable for optimisation.
    From: Zhou et al. "On the continuity of rotation representations in neural networks", CVPR, 2019.
    Effectively we use the first two column vectors of the rotation matrix as representation and apply the optimisation
    step to those vectors. Then, we use a Gram-Schmidt process to get normalised and orthogonal axes to construct
    the 3x3 rotation matrix.
    """
    @staticmethod
    def representation_for_optimisation(rot_mat):
        """ maps 3x3 rotation matrix to a 6d representation for optimisation """
        # use first two columns
        return rot_mat[:, :2].flatten()

    @staticmethod
    def rotation_matrix(mapped_rot):
        """ maps the 6d representation to 3x3 rotation matrix """
        # extract the vectors
        mapped_rot = mapped_rot.reshape(3, 2)
        a1 = mapped_rot[:, 0]
        a2 = mapped_rot[:, 1]

        # we search for the basis vectors b1, b2, b3
        # b1 is simply N(a1)
        b1 = a1 / torch.norm(a1)

        # b2 is N(a2 - (b1*a2)b1)
        b2 = a2 - b1.dot(a2) * b1
        b2 = b2 / torch.norm(b2)

        # b3 = b1 x b2
        b3 = b1.cross(b2)
        rot_mat = torch.stack([b1, b2, b3], dim=1)
        return rot_mat
