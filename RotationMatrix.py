import numpy as np
import numpy.linalg as l
from scipy.spatial.transform import Rotation as R
from functools import reduce
from constants import TOL
from typing import Tuple, List

def get_vec_orthogonal_to(v: np.array) -> np.array:
    """Given a non zero vector v return a vector orthorgonal to v"""
    res = np.zeros(3)
    for i in range(3):
        if v[i] == 0:
            res[i] = 1
            return res
    
    return np.array([v[1], -v[0], 0])

def get_uniform_random_rotation_scipy():
    r = R.random()
    return r.as_matrix()

def rotation_matrix_to_axis_angle_form(m: np.array)->Tuple[np.array, float]:
    """Given a 3 *3 rotation matrix M, Return a normalized axis of rotation and
    the counterclockwise angle of rotation about that axis"""
    e = l.eig(m)

    one_idx = None
    complex_eig = []
    complex_eig_v = []
    for i, v in enumerate(e.eigenvalues):
        if np.abs(v-1) < TOL:
            one_idx =i
        else:
            complex_eig.append(v)
            complex_eig_v.append(e.eigenvectors[:, i])
    
    assert one_idx is not None
    axis = e.eigenvectors[:, one_idx].real

    cross = np.cross(complex_eig_v[0], complex_eig_v[1]).imag
    if l.norm(cross-axis) < TOL:
        id_for_theta = 0
    else:
        id_for_theta = 1

    theta_1 = np.arccos(complex_eig[id_for_theta].real)
    theta_2 = np.pi-theta_1 + np.pi

    for theta in [theta_1, theta_2]:
        if np.abs(np.sin(theta) - complex_eig[id_for_theta].imag) < TOL:
            theta_sol = theta
            break

    return axis, theta_sol

def complete_to_right_handed_orthornomal_basis(v: np.array)-> List[np.array]:
    """Given a non-zero 3D vector return an a right handed orthonormal basis whose first vector is v"""
    v1 = get_vec_orthogonal_to(v)
    v1 = v1/l.norm(v1)
    v2 = np.cross(v, v1)
    return [v, v1, v2]

def axis_angle_to_rot_matrix(v: np.array, theta: float) -> np.array:
    """Given an axis and a counterclockwise angle of rotation theta around that axis, return the corresponding 3*3 rotation matrix"""
    v = v/l.norm(v)
    ortho_basis = complete_to_right_handed_orthornomal_basis(v)
    T = np.stack(ortho_basis)
    m =np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return reduce(np.dot, [T.transpose(), m, T])



if __name__ == '__main__':
    m = get_uniform_random_rotation_scipy()
    axis, theta = rotation_matrix_to_axis_angle_form(m)
    m_ = axis_angle_to_rot_matrix(axis, theta)

    print(l.norm(m-m_))
