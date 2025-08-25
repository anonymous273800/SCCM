import numpy as np


def define_plane_from_norm_vector_and_a_point(nv, point):
    """
       Define a plane using a normal vector and a point in the space.

       Parameters:
       nv (numpy array): A numpy array representing the normal vector of the targeted plane.
                         which resulted from the weighted avg from the two norm vectors
                         of the base and incremental models respectively.
       point (numpy array): A numpy array representing a point that lies on the plane.
                            This point (intersection point) helps determine the position of the plane in the space.

       Returns:
       numpy array: The coefficients of the new defined hyperplane.

       Ex. Find the equation of the plane out of the given norm vector n=<2,3,1>
            and the intersection point Q (1,1,1)
            Solving this mathematically should follow the following steps:
            1. Given nv = <2,3,1>
            2. Given Point on the target plane (intersection point) = (1,1,1)
            =>  Let an arbitrary point r = (x,y,z)
                define QP vector = r - b = <x - 1, y - 1, z - 1>
                nv . QP = <2,3,1> . <x-1, y-1, z-1>
                        = 2x - 2 + 3y -3 + z -1 = 0
                        = 2x + 3y + z -6 = 0 (the equation of the new plane)
            =>  Similarly calling the below method on
                nv = [2,3,1]
                point = [1,1,1]
                plane = define_plane_from_norm_vector_and_a_point(nv,point)
                print(plane)
                The result is:  [ 2  3  1 -6] which represents the coefficients of the target plane.
       """
    temp = -1 * sum(n * v for n, v in zip(nv, point))
    nv = np.append(nv, temp)
    return nv


