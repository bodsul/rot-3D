import numpy as np
from Quaternion import quaternion
from RotationMatrix import rotation_matrix_to_axis_angle_form, axis_angle_to_rot_matrix

def quaternion_to_rot_matrix(q: quaternion):
    axis, theta = quaternion.unit_quaternion_to_exp_form(q)
    return axis_angle_to_rot_matrix(axis, theta)

def quaternion_from_rot_matrix(m: np.array):
    axis, theta = rotation_matrix_to_axis_angle_form(m)
    return quaternion.unit_quaternion_from_exp_form(axis, theta)
if __name__=='__main__':
    r = quaternion.uniformly_random_unit_quaternion()
    m = quaternion_to_rot_matrix(r)
    r_ = quaternion_from_rot_matrix(m)

    print(r)
    print(r_)