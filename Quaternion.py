from utils import uniformly_random_point_on_n_sphere
from constants import TOL
import numpy as np
import numpy.linalg as l

class quaternion:
    def __init__(self, real: float, i: float, j: float, k: float) -> None:
        self.real = real
        self.i = i
        self.j = j
        self.k = k

    def __abs__(self) -> float:
        return self.real*self.real + self.i*self.i + self.j*self.j + self.k*self.k
    
    def __mul__(self, right):
        real = self.real * right.real - (self.i*right.i + self.j*right.j + self.k*right.k)
        i = self.j*right.k - self.k*right.j + self.real * right.i + self.i * right.real
        j = self.k*right.i - self.i*right.k+ self.real * right.j + self.j * right.real
        k = self.i*right.j - self.j*right.i + self.real * right.k + self.k * right.real
        return quaternion(real, i, j, k)
    
    def __str__(self):
        return f'{self.real} + {self.i}i + {self.j}j + {self.k}k'
    
    def __sub__(self, other):
        return quaternion(self.real-other.real, self.i-other.i, self.j-other.j, self.k-other.k)
    
    def __add__(self, other):
        return quaternion(self.real+other.real, self.i+other.i, self.j+other.j, self.k+other.k)

    def conjugate(self):
        return quaternion(self.real, -self.i, -self.j, -self.k)
    
    def Rotate(self, v):
        assert abs(abs(self)-1) < TOL

        v_q = quaternion(0, v[0], v[1], v[2])
        v_q_rot = self*v_q*self.conjugate()
        return np.array([v_q_rot.i, v_q_rot.j, v_q_rot.k])
    
    @classmethod
    def uniformly_random_unit_quaternion(cls):
        p = uniformly_random_point_on_n_sphere(3)
        return quaternion(p[0], p[1], p[2], p[3])
    
    @classmethod
    def unit_quaternion_to_exp_form(cls, q):
        assert abs(abs(q)-1) < TOL
        half_theta_1 = np.arccos(q.real)
        half_theta_2 = np.pi - half_theta_1 + np.pi
        theta_1 = half_theta_1*2
        theta_2 = half_theta_2/2

        for theta in [theta_1, theta_2]:
            axis = np.array([q.i/np.sin(theta/2), q.j/np.sin(theta/2), q.k/np.sin(theta/2)])
            if abs(l.norm(axis)-1) < TOL:
                return axis, theta
                
    @classmethod
    def unit_quaternion_from_exp_form(cls, axis, theta):
        assert abs(l.norm(axis)-1) < TOL
        cos_val = np.cos(theta/2)
        sin_val = np.sin(theta/2)
        return quaternion(cos_val, axis[0]*sin_val, axis[1]*sin_val, axis[2]*sin_val)
    

if __name__ == '__main__':
    q = quaternion(1, 1, 1, 1)

    r = quaternion.uniformly_random_unit_quaternion()

    imag = quaternion(0, 2, 1, 3)

    s = r.conjugate()*imag*r

    print(s, abs(s), abs(imag))

    axis, theta = quaternion.unit_quaternion_to_exp_form(r)

    print(axis, theta)

    

