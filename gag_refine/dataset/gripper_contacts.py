import numpy as np
import burg_toolkit as burg


supported_grippers = [
    burg.gripper.Franka
]


def get_gripper_contact_points(gripper):
    """
    Gives the expected contact points at the fingertips of the gripper.
    *Note* that this assumes the gripper's pose is identity matrix, i.e. grasp center is at origin, approach direction
    is the negative z-axis.

    Args:
        gripper: instance of a burg.gripper.GripperBase

    Returns:
        np.ndarray, Nx3 with N being the number of contact points
    """
    # need to get some private variables, it is safe to use, probably should change the interface in BURG toolkit
    bc = gripper._bullet_client
    gripper_scale_factor = gripper._gripper_size

    if isinstance(gripper, burg.gripper.Franka):
        # get the contact links
        contact_link_ids = gripper.get_contact_link_ids()
        contact_points = np.empty((len(contact_link_ids), 3))
        for i, contact_link in enumerate(contact_link_ids):
            link_state = bc.getLinkState(gripper.body_id, contact_link, computeForwardKinematics=True)
            contact_points[i] = link_state[4]

        # need to transform those points by moving them downwards to align with the fingertips and then moving them
        # closer to the origin, such that they are offset from the fingertips by a bit
        z_offset = 0.045 * gripper_scale_factor  # then they are roughly in the middle of the fingertip
        move_inwards_by = 0.004 * gripper_scale_factor  # then they are exactly at origin when open_scale = 0.1

        # apply z_offset
        contact_points = contact_points - np.asarray([0, 0, z_offset])

        # move points inwards to origin in x/y (which is the grasp center)
        move_direction = np.zeros_like(contact_points)
        move_direction[:, :2] = contact_points[:, :2]  # only x/y
        move_direction /= np.linalg.norm(move_direction, axis=-1)[:, None]
        moved_contact_points = contact_points - move_inwards_by * move_direction

        return moved_contact_points
    else:
        raise NotImplementedError(f'getting gripper contact points is not implemented for {type(gripper)}')
