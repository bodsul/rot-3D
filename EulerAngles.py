import numpy as np
from typing import Union, Tuple
from numpy import cos as cos, sin as sin, tan as tan, arctan2 as arctan2, arccos as arccos, arcsin as arcsin
from constants import TOL
import numpy.linalg as l
from Quaternion import quaternion
from Quat_Rot_Transformations import quaternion_to_rot_matrix, quaternion_from_rot_matrix

def x_y_z_euler_angles_to_rotation_matrix(roll, pitch, yaw):
    return np.array([[cos(pitch)*cos(yaw), sin(roll)*sin(pitch)*cos(yaw) - cos(roll)*sin(yaw), cos(roll)*sin(pitch)*cos(yaw) + sin(roll)*sin(yaw)],
                     [cos(pitch)*sin(yaw), sin(roll)*sin(pitch)*sin(yaw) + cos(roll)*cos(yaw), cos(roll)*sin(pitch)*sin(yaw) - sin(roll)*cos(yaw)],
                     [-sin(pitch), sin(roll)*cos(pitch), cos(roll)*cos(pitch)]
                     ])

def x_y_z_euler_angles_from_rotation_matrix(m) -> Union[Tuple[list, list, list], Tuple[float, float, float]]:
    if abs(m[2][0]- 1) > TOL and abs(m[2][0]+1) > TOL:
        pitch = [-arcsin(m[2][0]), np.pi + arcsin(m[2][0])]
        roll = [arctan2(m[2][1]/cos(p), m[2][2]/cos(p)) for p in pitch]
        yaw = [arctan2(m[1][0]/cos(p), m[0][0]/cos(p)) for p in pitch]
    else: # Gimbal lock
        if abs(m[2][0]- 1) < TOL:
            pitch = -np.pi/2
            yaw = 0
            roll = -yaw + arctan2(-m[0][1], -m[0][2])
        else:
            pitch = np.pi/2
            yaw = 0
            roll = yaw + arctan2(m[0][1], m[0][2])
    
    return roll, pitch, yaw


def x_y_z_euler_angles_to_unit_quaternion(roll, pitch, yaw):
    return quaternion(cos(yaw/2), 0, 0, sin(yaw/2))*quaternion(cos(pitch/2), 0, sin(pitch/2), 0)*quaternion(cos(roll/2), sin(roll/2), 0, 0)

def x_y_z_euler_angles_from_unit_quaternion(q):
    m = quaternion_to_rot_matrix(q)
    return x_y_z_euler_angles_from_rotation_matrix(m)


if __name__ == '__main__':
    [roll, pitch, yaw] = list(np.random.random(3)*2*np.pi)
    m = x_y_z_euler_angles_to_rotation_matrix(roll, pitch, yaw)
    q = x_y_z_euler_angles_to_unit_quaternion(roll, pitch, yaw)
    m_ = quaternion_to_rot_matrix(q)
    prod = m.dot(m.transpose())
    assert l.norm(prod -np.eye(3)) < TOL
    assert(l.norm(m-m_)) < TOL

    q = quaternion.uniformly_random_unit_quaternion()
    m = quaternion_to_rot_matrix(q)


    roll, pitch, yaw = x_y_z_euler_angles_from_rotation_matrix(m)
    if isinstance(roll, list):
        idx = np.random.random() > 0.5
        q_ = x_y_z_euler_angles_to_unit_quaternion(roll[idx], pitch[idx], yaw[idx])
    else:
        q_ = x_y_z_euler_angles_to_unit_quaternion(roll, pitch, yaw)

    # take care of q=-q antipodal identification
    assert(abs(q-q_) < TOL or abs(q+q_) < TOL)
